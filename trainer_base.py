# -*- coding: utf-8 -*-
"""
/////
"""
from tqdm import tqdm
import os
import logging
import torch
from models.PEDP import PEDP_model
from models.DiaMultiClass import DiaMultiClass
from models.DiaMultiDense import DiaMultiDense
from models.DiaSeq import DiaSeq
from evaluator.PEDP_evaluator import PEDPEvaluator
from dataloader.dbquery import DBQuery
from utils import to_device, seed_everything
from dataloader.datamanager import DataManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.best = -1
        if args.seq:
            self.net = DiaSeq(args, cfg).to(DEVICE)
        elif args.cls:
            self.net = DiaMultiClass(args, cfg).to(DEVICE)
        elif args.md:
            self.net = DiaMultiDense(args, cfg).to(DEVICE)
        elif args.pedp:
            self.net = PEDP_model(args, cfg).to(DEVICE)

        db = DBQuery(args.data_dir)
        datamanager = DataManager(args.data_dir, cfg)

        self.train_loader = datamanager.create_dataset_PEDP('train', args.batchsz, cfg, db, other_data=cfg.dataset_dir)
        self.valid_loader = datamanager.create_dataset_PEDP('valid', args.batchsz, cfg, db, other_data=cfg.dataset_dir)
        self.test_loader = datamanager.create_dataset_PEDP('test', args.batchsz, cfg, db, other_data=cfg.dataset_dir)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
        self.evaluator = PEDPEvaluator(args, cfg, robust=False)
        self.beta = args.beta_upper
        batch_num = len(self.train_loader)
        decay_batchs = batch_num * args.beta_decay
        self.real_decay = (args.beta_upper - args.beta_lower) / decay_batchs
        torch.set_num_threads(args.cpus)
        logging.info(self.net)
        logging.info('Planner initialization is done')

    def load_pretrained(self, load_dir=None):
        self.net.load_state_dict(torch.load(load_dir, map_location=torch.device(DEVICE)))

    def evaluate(self, epoch, rbs=False):
        self.net.eval()
        with torch.no_grad():
            iterator_agenda = enumerate(self.valid_loader)
            for batch_idx, data in iterator_agenda:
                s, a_tgt_pred, s_tgt_pred, s_tgt_pos, a_tgt_seq, a_full_tgt, a_tgt_pos = to_device(data)

                loss_plan, plan_act_tsr, loss_pred, pred_act_tsr, \
                loss_state, h_s_rec, loss_KL, \
                loss_term, plan_term_tsr, gold_term_tsr = self.net(
                    s,
                    a_tgt_pred,
                    0,
                    s_target_gold=s_tgt_pred,
                    s_target_pos=s_tgt_pos,
                    a_target_seq=a_tgt_seq,
                    train_type='valid',
                    a_target_full=a_full_tgt,
                    a_target_pos=a_tgt_pos
                )

                loss_all = self.args.pred_gamma * loss_pred \
                           + self.args.plan_gamma * (loss_plan + loss_term + loss_KL) \
                           + self.args.state_gamma * loss_state

                self.evaluator.step(loss_all.item(), loss_plan.item(), loss_pred.item(),
                                    plan_act_tsr, pred_act_tsr, data,
                                    loss_state.item(), h_s_rec, s_tgt_pred, loss_KL.item(),
                                    loss_term.item(), plan_term_tsr, gold_term_tsr)
                if batch_idx % self.args.log_interval == 0 and self.args.debug:
                    self.evaluator.summary('TEST', epoch * len(self.train_loader) + batch_idx, self.net)
            val_success, val_f1 = self.evaluator.summary('VALID', epoch)

        with torch.no_grad():
            iterator_agenda = tqdm(enumerate(self.test_loader)) if self.args.retest else enumerate(self.test_loader)
            for batch_idx, data in iterator_agenda:
                s, a_tgt_pred, s_tgt_pred, s_tgt_pos, a_tgt_seq, a_full_tgt, a_tgt_pos = to_device(data)

                loss_plan, plan_act_tsr, loss_pred, pred_act_tsr, \
                loss_state, h_s_rec, loss_KL, \
                loss_term, plan_term_tsr, gold_term_tsr = self.net(
                    s,
                    a_tgt_pred,
                    0,
                    s_target_gold=s_tgt_pred,
                    s_target_pos=s_tgt_pos,
                    a_target_seq=a_tgt_seq,
                    train_type='test',
                    a_target_full=a_full_tgt,
                    a_target_pos=a_tgt_pos
                )

                loss_all = self.args.pred_gamma * loss_pred \
                           + self.args.plan_gamma * (loss_plan + loss_term + loss_KL) \
                           + self.args.state_gamma * loss_state

                self.evaluator.step(loss_all.item(), loss_plan.item(), loss_pred.item(),
                                    plan_act_tsr, pred_act_tsr, data,
                                    loss_state.item(), h_s_rec, s_tgt_pred, loss_KL.item(),
                                    loss_term.item(), plan_term_tsr, gold_term_tsr)
            net_eval = self.net if (epoch + 1) % 1 == 0 or rbs else None
            test_success, test_f1 = self.evaluator.summary('TEST', epoch, net_eval)
        if not self.args.test:
            save_path = self.args.checkpoint_dir
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save({'epoch': epoch + 1,
                        'state_dict': self.net.state_dict(),
                        'best_f1': test_success}, save_path + '/{}.pth.tar'.format(epoch))
            if test_f1 > self.best:
                self.best = test_f1
                torch.save({'epoch': epoch + 1,
                            'state_dict': self.net.state_dict(),
                            'best_f1': test_success}, save_path + '/best.pth.tar')

    def imitate(self, epoch):
        seed_everything(seed=self.args.seed)
        self.net.train()
        for batch_idx, data in enumerate(self.train_loader):
            self.optim.zero_grad()
            s, a_tgt_pred, s_tgt_pred, s_tgt_pos, a_tgt_seq, a_full_tgt, a_tgt_pos = to_device(data)

            loss_plan, plan_act_tsr, loss_pred, pred_act_tsr, \
            loss_state, h_s_rec, loss_KL, \
            loss_term, plan_term_tsr, gold_term_tsr = self.net(
                s,
                a_tgt_pred,
                self.beta,
                s_target_gold=s_tgt_pred,
                s_target_pos=s_tgt_pos,
                a_target_seq=a_tgt_seq,
                train_type='train',
                a_target_full=a_full_tgt,
                a_target_pos=a_tgt_pos
            )

            loss_all = self.args.pred_gamma * loss_pred \
                       + self.args.plan_gamma * (loss_plan + loss_term + loss_KL) \
                       + self.args.state_gamma * loss_state
            self.evaluator.step(loss_all.item(), loss_plan.item(), loss_pred.item(),
                                plan_act_tsr, pred_act_tsr, data,
                                loss_state.item(), h_s_rec, s_tgt_pred, loss_KL.item(),
                                loss_term.item(), plan_term_tsr, gold_term_tsr)
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optim.step()
            if self.beta > self.args.beta_lower:
                self.beta -= self.real_decay
            else:
                self.beta = self.args.beta_lower
            if batch_idx % self.args.log_interval == 0:
                self.evaluator.summary('TRAIN', epoch * len(self.train_loader) + batch_idx)
                if self.args.debug:
                    break

        if epoch >= 0:
            self.evaluate(epoch)
