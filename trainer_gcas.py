# -*- coding: utf-8 -*-
"""
/////
"""
import os
import logging
import torch
from models.GCAS import gCAS
from evaluator.GCAS_evaluator import GCASEvaluator
from dataloader.dbquery import DBQuery
from utils import to_device, seed_everything
from dataloader.datamanager import DataManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainerGCAS:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.best = -1

        di_lst = list(set([x.split('-')[1] for x in cfg.da]))
        cfg.di2idx = dict((s, i) for i, s in enumerate(di_lst))
        cfg.idx2di = dict((v, k) for k, v in cfg.di2idx.items())

        sv_lst = list(set([x.split('-')[0] + '-' + '-'.join(x.split('-')[2:]) for x in cfg.da]))
        cfg.sv2idx = dict((s, i) for i, s in enumerate(sv_lst))
        cfg.idx2sv = dict((v, k) for k, v in cfg.sv2idx.items())
        cfg.gcas_c_dim = 2
        cfg.gcas_a_dim = len(di_lst)
        cfg.gcas_s_dim = len(sv_lst)

        self.net = gCAS(args, cfg).to(DEVICE)

        db = DBQuery(args.data_dir)
        datamanager = DataManager(args.data_dir, cfg)
        self.train_loader = datamanager.create_dataset_gcas('train', args.batchsz, cfg, db)
        self.valid_loader = datamanager.create_dataset_gcas('valid', args.batchsz, cfg, db)
        self.test_loader = datamanager.create_dataset_gcas('test', args.batchsz, cfg, db)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-5)
        self.evaluator = GCASEvaluator(args, cfg)
        torch.set_num_threads(args.cpus)
        logging.info(self.net)

    def load_pretrained(self, load_dir=None):
        best_pol_pkl_dir = self.args.checkpoint_dir + load_dir
        self.net.load_state_dict(torch.load(best_pol_pkl_dir))
        if os.path.exists(best_pol_pkl_dir):
            best_pol_pkl = torch.load(best_pol_pkl_dir)
            self.net.load_state_dict(best_pol_pkl['state_dict'])
            logging.info('*' * 100)
            logging.info('World loaded: epoch {} | Criteria {}'.format(best_pol_pkl['epoch'], best_pol_pkl['best_f1']))
        else:
            logging.info('World loading failed, no such file: {}'.format(best_pol_pkl_dir))

    def evaluate(self, epoch):
        self.net.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_loader):
                s, c_target_gold, i_target_gold, s_target_gold, a_target_gold = to_device(data)
                loss_c, loss_a, loss_s, pred_act_tsr = self.net(s, c_target_gold, i_target_gold, s_target_gold, 'val')
                loss_pred = loss_c + loss_a + loss_s
                self.evaluator.step(loss_c.item(), loss_a.item(), loss_s.item(), loss_pred.item(), pred_act_tsr,
                                    a_target_gold)
                if batch_idx % self.args.log_interval == 0 and self.args.debug:
                    self.evaluator.summary('TEST', epoch * len(self.train_loader) + batch_idx, self.net)
            _ = self.evaluator.summary('VALID', epoch)
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                s, c_target_gold, i_target_gold, s_target_gold, a_target_gold = to_device(data)
                loss_c, loss_a, loss_s, pred_act_tsr = self.net(s, c_target_gold, i_target_gold, s_target_gold, 'test')
                loss_pred = loss_c + loss_a + loss_s
                self.evaluator.step(loss_c.item(), loss_a.item(), loss_s.item(), loss_pred.item(), pred_act_tsr,
                                    a_target_gold)
            net_eval = self.net if (epoch + 1) % 1 == 0 else None
            test_success = self.evaluator.summary('TEST', epoch, net_eval)
        if not self.args.test:
            save_path = self.args.checkpoint_dir
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            torch.save({'epoch': epoch + 1,
                        'state_dict': self.net.state_dict(),
                        'best_f1': test_success}, save_path + '/{}.pth.tar'.format(epoch))

    def imitate(self, epoch):
        seed_everything(seed=self.args.seed)
        self.net.train()
        for batch_idx, data in enumerate(self.train_loader):
            self.optim.zero_grad()
            s, c_target_gold, i_target_gold, s_target_gold, a_target_gold = to_device(data)
            loss_c, loss_a, loss_s, pred_act_tsr = self.net(s, c_target_gold, i_target_gold, s_target_gold)
            loss_pred = loss_c + loss_a + loss_s
            loss_pred.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.clip)
            self.optim.step()
            self.evaluator.step(loss_c.item(), loss_a.item(), loss_s.item(), loss_pred.item(), pred_act_tsr, a_target_gold)
            if batch_idx % self.args.log_interval == 0:
                _ = self.evaluator.summary('TRAIN', epoch * len(self.train_loader) + batch_idx)
                if self.args.debug:
                    break

        if epoch >= 0:
            self.evaluate(epoch)

