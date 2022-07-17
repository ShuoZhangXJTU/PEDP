"""
 2021.03.03 by Shuo Zhang
"""
import sklearn.metrics as metrics
from numpy import mean
import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from evaluator.agenda_evaluator import AgendaEvaluator
from sklearn.preprocessing import MultiLabelBinarizer


def evaluate(net, real, diff):
    counter_match = 0
    ttl_counter = 0
    for raw_net, raw_real in zip(net, real):
        if torch.equal(raw_net, raw_real):
            counter_match += 1
        ttl_counter += 1
    match_rate = counter_match / ttl_counter

    if diff:
        pred = net.gt(0.5).long().detach().cpu().numpy()
        true = real.cpu().numpy()
    precision = metrics.precision_score(true, pred, average='samples', zero_division=0)
    recall = metrics.recall_score(true, pred, average='samples', zero_division=0)
    F1 = metrics.f1_score(true, pred, average='samples', zero_division=0)

    return F1, precision, recall, match_rate


def multi_evaluate(net_plan, real_pred_seq, net_pred):
    # -- now we eval like multi-hot vec
    # -- ignore 0, 1, 2, 171, 172 as sos, pad, eos, sos and eos
    # ---- real
    # true_dim = 86  # multiwoz 166
    # print(net_pred.shape)
    true_dim = net_pred.shape[-1]
    mlb = MultiLabelBinarizer(classes=list(range(true_dim)))

    real_plan = mlb.fit_transform(real_pred_seq.tolist())
    real_plan = torch.from_numpy(real_plan)

    real_pred = mlb.fit_transform(real_pred_seq.tolist())
    real_pred = torch.from_numpy(real_pred)

    # ---- pred
    net_plan, net_pred = net_plan, net_pred

    rst = []
    for net, real in [(net_plan, real_plan), (net_pred, real_pred)]:

        true = real.cpu().numpy()
        pred = net.detach().cpu().numpy()
        avg_ = 'samples'
        # import ipdb; ipdb.set_trace()
        precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
        recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
        F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
        rst += [F1, precision, recall]
    return rst


def term_evaluate(pred, real):
    true = real.cpu().numpy()
    pred = pred.detach().cpu().numpy()
    avg_ = 'macro'
    precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
    recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
    F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
    return [F1, precision, recall]


def multi_evaluate_old(net_plan, real_pred_seq, net_pred, curriculum=0):
    if net_plan is not None:
        # -- now we eval like multi-hot vec
        # -- ignore 0, 1, 2, 171, 172 as sos, pad, eos, sos and eos
        # ---- real
        mlb = MultiLabelBinarizer(classes=list(range(171)))

        real_plan = mlb.fit_transform(real_pred_seq[:, :20].tolist())
        real_plan = torch.from_numpy(real_plan)[:, 3:-1]

        real_pred = mlb.fit_transform(real_pred_seq[:, 20:].tolist())
        real_pred = torch.from_numpy(real_pred)[:, 3:-1]

        # ---- pred
        net_plan, net_pred = net_plan[:, 3:-1], net_pred[:, 3:-1]

        rst = []
        for net, real in [(net_plan, real_plan), (net_pred, real_pred)]:

            true = real.cpu().numpy()
            pred = net.detach().cpu().numpy()
            avg_ = 'samples'
            precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
            recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
            F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
            rst += [F1, precision, recall]
    else:
        mlb = MultiLabelBinarizer(classes=list(range(171)))
        real_pred = mlb.fit_transform(real_pred_seq[:, 20:].tolist())
        real_pred = torch.from_numpy(real_pred)[:, 3:-1]
        # ---- pred
        net_pred = net_pred[:, 3:-1]
        # import ipdb; ipdb.set_trace()
        rst = [0,0,0]
        for net, real in [(net_pred, real_pred)]:
            true = real.cpu().numpy()
            pred = net.detach().cpu().numpy()
            avg_ = 'samples'
            precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
            recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
            F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
            rst += [F1, precision, recall]

    return rst


class PEDPEvaluator:
    def __init__(self, args, cfg, robust=True):
        self.result_buffer = []
        tb_path = args.tb_dir
        if not os.path.exists(tb_path):
            os.mkdir(tb_path)
        self.args = args
        self.writer = SummaryWriter(tb_path)
        self.robust = robust
        if robust:
            self.robust_evaluator = AgendaEvaluator(args, cfg)

    def write(self, title, value, step):
        self.writer.add_text(title, str(value), step)

    def summary(self, mode, epoch, policy=None, tune=False):
        loss_plan, loss_pred, loss_all, plan_f1, plan_prc, plan_rcl,\
        pred_f1, pred_prc, pred_rcl, loss_state, state_f1, state_prc, state_rcl, \
        loss_kl, loss_term, term_f1, term_prc, term_rcl = [mean([x[i] for x in self.result_buffer]) for i in range(18)]

        self.writer.add_scalar('{}-std/Loss-Plan'.format(mode), loss_plan, epoch)
        self.writer.add_scalar('{}-std/Loss-Pred'.format(mode), loss_pred, epoch)
        self.writer.add_scalar('{}-std/Loss-ALL'.format(mode), loss_all, epoch)
        self.writer.add_scalar('{}-std/Plan-F1'.format(mode), plan_f1, epoch)
        self.writer.add_scalar('{}-std/Plan-Precision'.format(mode), plan_prc, epoch)
        self.writer.add_scalar('{}-std/Plan-Recall'.format(mode), plan_rcl, epoch)
        self.writer.add_scalar('{}-std/Pred-F1'.format(mode), pred_f1, epoch)
        self.writer.add_scalar('{}-std/Pred-Precision'.format(mode), pred_prc, epoch)
        self.writer.add_scalar('{}-std/Pred-Recall'.format(mode), pred_rcl, epoch)
        self.writer.add_scalar('{}-std/Loss-State'.format(mode), loss_state, epoch)
        self.writer.add_scalar('{}-std/State-F1'.format(mode), state_f1, epoch)
        self.writer.add_scalar('{}-std/State-Precision'.format(mode), state_prc, epoch)
        self.writer.add_scalar('{}-std/State-Recall'.format(mode), state_rcl, epoch)
        self.writer.add_scalar('{}-std/Loss-KL'.format(mode), loss_kl, epoch)
        self.writer.add_scalar('{}-std/Loss-Term'.format(mode), loss_term, epoch)
        self.writer.add_scalar('{}-std/Term-F1'.format(mode), term_f1, epoch)
        self.writer.add_scalar('{}-std/Term-Precision'.format(mode), term_prc, epoch)
        self.writer.add_scalar('{}-std/Term-Recall'.format(mode), term_rcl, epoch)

        if mode == 'test':
            logging.info('=' * 36)
        logging.info('{} | step {} | Loss {:.6} | Loss-plan {:.6} | Loss-pred {:.6} | Loss-state {:.6} | Loss-KL {:.6} '
                     '| Loss-Term {:.6} | | Plan F1 {:.4} | Plan P {:.4} | Plan R {:.4}'
                     '| Pred F1 {:.4} | Pred P {:.4} | Pred R {:.4}'
                     '| State F1 {:.4} | State P {:.4} | State R {:.4}'
                     '| Term F1 {:.4} | Term P {:.4} | Term R {:.4}'.format(mode, epoch, loss_all, loss_plan,
                                                                            loss_pred, loss_state, loss_kl, loss_term,
                                                                            plan_f1, plan_prc, plan_rcl,
                                                                            pred_f1, pred_prc, pred_rcl,
                                                                            state_f1, state_prc, state_rcl,
                                                                            term_f1, term_prc, term_rcl))
        success_here = 0.
        if policy:
            if self.robust:
                R_inform, F1_inform, success, reward, turn, match = self.robust_evaluator.evaluate(policy)
            else:
                R_inform, F1_inform, success, reward, turn, match = 0, 0, 0, 0, 0, 0
            self.writer.add_scalar('{}-rbs/Inform Recall'.format(mode), R_inform, epoch)
            self.writer.add_scalar('{}-rbs/Inform F1'.format(mode), F1_inform, epoch)
            self.writer.add_scalar('{}-rbs/Success'.format(mode), success, epoch)
            self.writer.add_scalar('{}-rbs/Reward'.format(mode), reward, epoch)
            self.writer.add_scalar('{}-rbs/Turn'.format(mode), turn, epoch)
            self.writer.add_scalar('{}-rbs/Match'.format(mode), match, epoch)
            success_here += success

        if tune:
            return loss_plan, loss_pred, loss_all, plan_f1, plan_prc, plan_rcl,\
                   pred_f1, pred_prc, pred_rcl, loss_state, state_f1, state_prc, state_rcl,\
                   loss_kl, loss_term, term_f1, term_prc, term_rcl, \
                   R_inform, F1_inform, success, reward, turn, match

        self.result_buffer = []
        return success_here, pred_f1

    def step(self, loss_a, loss_plan, loss_pred, plan_act_tsr, pred_act_tsr, data,
             state_loss, h_s_rec, h_s_tgt, loss_KL, loss_term, plan_term_tsr, gold_term_tsr):
        rst = multi_evaluate(plan_act_tsr, data[1], pred_act_tsr)
        if not loss_plan:
            loss_plan = 0
        rst += [state_loss]
        if h_s_rec is not None:
            pred = torch.sigmoid(h_s_rec).ge(0.5).long().detach().cpu().numpy()
            true = h_s_tgt.detach().cpu().numpy()
            avg_ = 'samples'
            precision = metrics.precision_score(true, pred, average=avg_, zero_division=0)
            recall = metrics.recall_score(true, pred, average=avg_, zero_division=0)
            F1 = metrics.f1_score(true, pred, average=avg_, zero_division=0)
            rst += [F1, precision, recall]
        else:
            rst += [0, 0, 0]
        rst += [loss_KL]

        rst += [loss_term]
        rst += term_evaluate(plan_term_tsr, gold_term_tsr)

        self.result_buffer.append([loss_plan, loss_pred, loss_a] + rst)
