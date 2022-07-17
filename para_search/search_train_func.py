# -*- coding: utf-8 -*-
import torch
import os
import argparse
from ray import tune
from models.PEDP_cls_best_0822 import PEDP_cls_best
from models.PEDP_seq_best_0822 import PEDP_seq_best
from models.PEDP import PEDP_model
from evaluator.PEDP_evaluator import PEDPEvaluator
from dataloader.dbquery import DBQuery
from utils import to_device, seed_everything
from dataloader.datamanager import DataManager
from config import MultiWozConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_for_turn(args, checkpoint_dir=None):
    cfg = MultiWozConfig()
    args = argparse.Namespace(**args)
    args.h_dim = int(args.h_dim)
    args.z_dim = int(args.z_dim)
    args.hz_dim = int(args.hz_dim)

    cfg.temperature = args.temperature
    cfg.data_ratio = args.data_ratio
    cfg.dropout = args.dropout
    cfg.h_dim = args.h_dim
    cfg.full_data = args.full_data

    db = DBQuery(args.data_dir)
    datamanager = DataManager(args.data_dir, cfg)
    beta = args.beta_upper
    net = PEDP_model(args, cfg).to(DEVICE)

    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    evaluator = PEDPEvaluator(args, cfg)
    train_loader = datamanager.create_dataset_PEDP('train', args.batchsz, cfg, db, args.full_data)
    valid_loader = datamanager.create_dataset_PEDP('valid', args.batchsz, cfg, db, args.full_data)
    # test_loader = datamanager.create_dataset_PEDP('test', args.batchsz, cfg, db)
    batch_num = len(train_loader)
    decay_batchs = batch_num * args.beta_decay
    real_decay = (args.beta_upper - args.beta_lower) / decay_batchs

    for epoch in range(80):
        seed_everything(6666)
        net.train()
        for batch_idx, data in enumerate(train_loader):
            optim.zero_grad()
            s, a_tgt_pred, s_tgt_pred, s_tgt_pos, a_tgt_seq, a_full_tgt, a_tgt_pos = to_device(data)

            loss_plan, plan_act_tsr, loss_pred, pred_act_tsr, \
            loss_state, h_s_rec, loss_KL, \
            loss_term, plan_term_tsr, gold_term_tsr = net(
                s,
                a_tgt_pred,
                beta,
                s_target_gold=s_tgt_pred,
                s_target_pos=s_tgt_pos,
                a_target_seq=a_tgt_seq,
                train_type='train',
                a_target_full=a_full_tgt,
                a_target_pos=a_tgt_pos
            )

            loss_all = args.pred_gamma * loss_pred \
                       + args.plan_gamma * loss_plan \
                       + args.kl_gamma * loss_KL \
                       + args.state_gamma * loss_state \
                       + args.term_gamma * loss_term

            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optim.step()
            if beta > args.beta_lower:
                beta -= real_decay
            else:
                beta = args.beta_lower

        net.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(valid_loader):
                s, a_tgt_pred, s_tgt_pred, s_tgt_pos, a_tgt_seq, a_full_tgt, a_tgt_pos = to_device(data)

                loss_plan, plan_act_tsr, loss_pred, pred_act_tsr, \
                loss_state, h_s_rec, loss_KL, \
                loss_term, plan_term_tsr, gold_term_tsr = net(
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

                loss_all = args.pred_gamma * loss_pred \
                           + args.plan_gamma * loss_plan \
                           + args.kl_gamma * loss_KL \
                           + args.state_gamma * loss_state \
                           + args.term_gamma * loss_term

                evaluator.step(loss_all.item(), loss_plan.item(), loss_pred.item(),
                               plan_act_tsr, pred_act_tsr, data,
                               loss_state.item(), h_s_rec, s_tgt_pred, loss_KL.item(),
                               loss_term.item(), plan_term_tsr, gold_term_tsr)
        loss_plan, loss_pred, loss_all, plan_f1, plan_prc, plan_rcl, \
        pred_f1, pred_prc, pred_rcl, loss_state, state_f1, state_prc, state_rcl, \
        loss_kl, loss_term, term_f1, term_prc, term_rcl, \
        R_inform, F1_inform, success, reward, turn, match = evaluator.summary('valid', epoch, net, tune=True)

        with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict()), path)

        tune.report(success=success, inform_f1=F1_inform, inform_R=R_inform, match=match, turn=turn,
                    state_loss=loss_state, state_F1=state_f1, state_p=state_prc, state_r=state_rcl,
                    plan_loss=loss_plan, plan_F1=plan_f1, plan_p=plan_prc, plan_r=plan_rcl,
                    term_loss=loss_term, term_F1=term_f1, term_p=term_prc, term_r=term_rcl,
                    pred_loss=loss_pred, pred_F1=pred_f1, pred_p=pred_prc, pred_r=pred_rcl,
                    all_loss=loss_all, )
