import sklearn.metrics as metrics
from numpy import mean
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from evaluator.agenda_evaluator import AgendaEvaluator


class GCASEvaluator:
    def __init__(self, args, cfg):
        self.result_buffer = []
        tb_path = args.tb_dir
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        self.args = args
        print(tb_path)
        self.writer = SummaryWriter(tb_path)
        # self.robust_evaluator = AgendaEvaluator(args, cfg)

    def summary(self, mode, epoch, policy=None):
        self.writer.add_scalar('{}-std/LossC'.format(mode),
                               mean([x[0] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/LossA'.format(mode),
                               mean([x[1] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/LossS'.format(mode),
                               mean([x[2] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/LossALL'.format(mode),
                               mean([x[3] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/Pred-F1'.format(mode),
                               mean([x[4] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/Pred-Precision'.format(mode),
                               mean([x[5] for x in self.result_buffer]), epoch)
        self.writer.add_scalar('{}-std/Pred-Recall'.format(mode),
                               mean([x[6] for x in self.result_buffer]), epoch)
        if mode == 'test':
            logging.info('=' * 36)
        logging.info('{} | step {} | LossC {:.6} | LossA {:.6} |LossS {:.6} | Loss {:.6} |'
                     'Pred F1 {:.4} | Pred P {:.4} | Pred R {:.4}'.format(mode, epoch,
                                                                          mean([x[0] for x in self.result_buffer]),
                                                                          mean([x[1] for x in self.result_buffer]),
                                                                          mean([x[2] for x in self.result_buffer]),
                                                                          mean([x[3] for x in self.result_buffer]),
                                                                          mean([x[4] for x in self.result_buffer]),
                                                                          mean([x[5] for x in self.result_buffer]),
                                                                          mean([x[6] for x in self.result_buffer]),
                                                                        ))
        success_here = 0.
        if policy:
            R_inform, F1_inform, success, reward, turn, match = 0, 0, 0, 0, 0, 0
            # R_inform, F1_inform, success, reward, turn, match = self.robust_evaluator.evaluate(policy)
            self.writer.add_scalar('{}-rbs/Inform Recall'.format(mode), R_inform, epoch)
            self.writer.add_scalar('{}-rbs/Inform F1'.format(mode), F1_inform, epoch)
            self.writer.add_scalar('{}-rbs/Success'.format(mode), success, epoch)
            self.writer.add_scalar('{}-rbs/Reward'.format(mode), reward, epoch)
            self.writer.add_scalar('{}-rbs/Turn'.format(mode), turn, epoch)
            self.writer.add_scalar('{}-rbs/Match'.format(mode), match, epoch)
            success_here += success
        self.result_buffer = []
        return success_here

    def step(self, loss_c, loss_a, loss_s, loss, pred_act_tsr, tgt_act_tsr):
        rst = [loss_c, loss_a, loss_s, loss]
        pred = pred_act_tsr.detach().cpu().numpy()
        true = tgt_act_tsr.cpu().numpy()
        avg_ = 'samples'
        precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
        recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
        F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
        rst += [F1, precision, recall]
        self.result_buffer.append(rst)
