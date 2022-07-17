# -*- coding: utf-8 -*-
"""
/////
"""
import os
import time
import numpy as np
from tqdm import tqdm
from random import choice
import random
import json
import logging
import torch
import torch.utils.data as data
from copy import deepcopy
from utils import init_session, init_goal, state_vectorize, action_vectorize, domain_vectorize, action_seq, pad_to
import itertools

def expand_da(meta):
    for k, v in meta.items():
        domain, intent = k.split('-')
        if intent.lower() == "request":
            for pair in v:
                pair.insert(1, '?')
        else:
            counter = {}
            for pair in v:
                if pair[0] == 'none':
                    pair.insert(1, 'none')
                else:
                    if pair[0] in counter:
                        counter[pair[0]] += 1
                    else:
                        counter[pair[0]] = 1
                    pair.insert(1, str(counter[pair[0]]))


class DataManager:
    """Offline data manager"""
    def __init__(self, data_dir, cfg):
        self.data = {}
        self.goal = {}
        self.cfg = cfg
        self.pol_dataset = dict()
        self.data_dir = data_dir
        self.data_dir_new = data_dir + '/processed_data'
        self.a_seq_lst = None
        self.loaded_data = None
        if os.path.exists(self.data_dir_new):
            logging.info('Load processed data file')
            for part in ['train', 'valid', 'test']:
                with open(self.data_dir_new + '/' + part + '.json', 'r') as f:
                    self.data[part] = json.load(f)
                with open(self.data_dir_new + '/' + part + '_goal.json', 'r') as f:
                    self.goal[part] = json.load(f)
        else:
            from .dbquery import DBQuery
            db = DBQuery(data_dir)
            logging.info('Start pre-processing the dataset')
            self._build_data(data_dir, self.data_dir_new, cfg, db)
            
    def _build_data(self, data_dir, data_dir_new, cfg, db, do_check=True):
        data_filename = data_dir + '/' + cfg.data_file
        with open(data_filename, 'r') as f:
            origin_data = json.load(f)
        for part in ['train', 'valid', 'test']:
            self.data[part] = []
            self.goal[part] = {}
        valList = []
        with open(data_dir + '/' + cfg.val_file) as f:
            for line in f:
                valList.append(line.split('.')[0])
        testList = []
        with open(data_dir + '/' + cfg.test_file) as f:
            for line in f:
                testList.append(line.split('.')[0])
        for k_sess in tqdm(origin_data):
            sess = origin_data[k_sess]
            if k_sess in valList:
                part = 'valid'
            elif k_sess in testList:
                part = 'test'
            else:
                part = 'train'
            turn_data, session_data = init_session(k_sess, cfg)
            init_goal(session_data, sess['goal'], cfg)
            self.goal[part][k_sess] = session_data
            belief_state = turn_data['belief_state']
            for i, turn in enumerate(sess['log']):
                book_status = turn['metadata']
                print(turn['dialog_act'])
                # if len(book_status) == 0: continue
                # for domain in self.cfg.belief_domains:
                #     if book_status[domain]['book']['booked']:
                #         print(turn['metadata'])

        import ipdb;ipdb.set_trace()
        for k_sess in tqdm(origin_data):
            sess = origin_data[k_sess]
            if k_sess in valList:
                part = 'valid'
            elif k_sess in testList:
                part = 'test'
            else:
                part = 'train'

            turn_data, session_data = init_session(k_sess, cfg)
            init_goal(session_data, sess['goal'], cfg)
            self.goal[part][k_sess] = session_data
            belief_state = turn_data['belief_state']

            for i, turn in enumerate(sess['log']):
                turn_data['others']['turn'] = i
                turn_data['others']['terminal'] = i + 2 >= len(sess['log'])
                da_origin = turn['dialog_act']
                expand_da(da_origin)
                turn_data['belief_state'] = deepcopy(belief_state)  # from previous turn

                if i % 2 == 0:  # user
                    if 'last_sys_action' in turn_data:
                        turn_data['history']['sys'] = dict(turn_data['history']['sys'], **turn_data['last_sys_action'])
                        del(turn_data['last_sys_action'])
                    turn_data['last_user_action'] = deepcopy(turn_data['user_action'])
                    turn_data['user_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        if _intent == 'thank':
                            _intent = 'welcome'
                            _domint = _domain+'-'+_intent
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da_usr:
                                turn_data['user_action'][_da] = _value
                                if _intent == 'inform':
                                    inform_da = _domain+'-'+_slot+'-1'
                                    if inform_da in cfg.inform_da:
                                        belief_state['inform'][_domain][_slot] = _value
                                elif _intent == 'request':
                                    request_da = _domain+'-'+_slot
                                    if request_da in cfg.request_da:
                                        belief_state['request'][_domain].add(_slot)
                else:  # sys
                    if 'last_user_action' in turn_data:
                        turn_data['history']['user'] = dict(turn_data['history']['user'], **turn_data['last_user_action'])
                        del(turn_data['last_user_action'])
                    turn_data['last_sys_action'] = deepcopy(turn_data['sys_action'])
                    turn_data['sys_action'] = dict()
                    for domint in da_origin:
                        domain_intent = da_origin[domint]
                        _domint = domint.lower()
                        _domain, _intent = _domint.split('-')
                        for slot, p, value in domain_intent:
                            _slot = slot.lower()
                            _value = value.strip()
                            _da = '-'.join((_domint, _slot, p))
                            if _da in cfg.da:
                                turn_data['sys_action'][_da] = _value
                                if _intent == 'inform' and _domain in belief_state['request']:
                                    belief_state['request'][_domain].discard(_slot)
                                elif _intent == 'book' and _slot == 'ref':
                                    for domain in belief_state['request']:
                                        if _slot in belief_state['request'][domain]:
                                            belief_state['request'][domain].remove(_slot)
                                            break

                    book_status = turn['metadata']
                    for domain in cfg.belief_domains:
                        if book_status[domain]['book']['booked']:
                            entity = book_status[domain]['book']['booked'][0]
                            if domain == 'taxi':
                                belief_state['booked'][domain] = 'booked'
                            elif domain == 'train':
                                found = db.query(domain, [('trainID', entity['trainID'])])
                                belief_state['booked'][domain] = found[0]['ref']
                            else:
                                found = db.query(domain, [('name', entity['name'])])
                                belief_state['booked'][domain] = found[0]['ref']
                if i + 1 == len(sess['log']):
                    turn_data['next_belief_state'] = belief_state
                self.data[part].append(deepcopy(turn_data))

        def _set_default(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError
        os.makedirs(data_dir_new)
        for part in ['train', 'valid', 'test']:
            with open(data_dir_new + '/' + part + '.json', 'w') as f:
                self.data[part] = json.dumps(self.data[part], default=_set_default)
                f.write(self.data[part])
                self.data[part] = json.loads(self.data[part])
            with open(data_dir_new + '/' + part + '_goal.json', 'w') as f:
                self.goal[part] = json.dumps(self.goal[part], default=_set_default)
                f.write(self.goal[part])
                self.goal[part] = json.loads(self.goal[part])
        
    def create_dataset(self, part, file_dir, cfg, db):
        datas = self.data[part]
        goals = self.goal[part]
        s, a, next_s, a_seq = [], [], [], []
        d_m = []
        logging.info('total {} samples'.format(len(datas)))
        sa2label = dict()
        for idx, turn_data in tqdm(enumerate(datas)):
            if turn_data['others']['turn'] % 2 == 0:
                continue
            turn_data['user_goal'] = goals[turn_data['others']['session_id']]
            aug_action = None
            state_tsr = torch.Tensor(state_vectorize(turn_data, cfg, db, True))
            s.append(state_tsr)
            act_tsr = torch.Tensor(action_vectorize(turn_data, cfg, provided_act=aug_action))
            a.append(act_tsr)
            d_m.append(torch.Tensor(domain_vectorize(turn_data, cfg)))
            a_seq.append(torch.Tensor(action_seq(turn_data, cfg, provided_act=aug_action)))
            if not int(turn_data['others']['terminal']):
                next_s.append(torch.Tensor(state_vectorize(datas[idx+2], cfg, db, True)))
            else:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['last_sys_action'] = next_turn_data['sys_action']
                next_turn_data['sys_action'] = {}
                next_turn_data['belief_state'] = next_turn_data['next_belief_state']
                next_s.append(torch.Tensor(state_vectorize(next_turn_data, cfg, db, True)))
        torch.save((s, a, next_s, d_m, a_seq), file_dir)

    def create_dataset_PEDP(self, part, batchsz, cfg, db, full_data=False, other_data=None):
        t0 = time.time()
        if other_data is None:
            logging.info('=' * 100)
            load_file_dir = self.data_dir_new + '/' + part + '.pt'
            if not os.path.exists(load_file_dir):
                logging.info('MultiWOZ | Create Dataset: {}'.format(load_file_dir))
                self.create_dataset(part, load_file_dir, cfg, db)

            s, a, next_s, _ = torch.load(load_file_dir)
            logging.info('Source Loaded: Total {} samples'.format(len(a)))
            n_s, n_next_s, a_seq, s_pos_seq = [], [], [], []
            a_seq_next, a_pos_seq = [], []
            dist_dict = dict()
            cur_id = 0

            turn_id_list = []
            for i, turn_data in enumerate(self.data[part]):
                if i % 2 == 1:
                    turn_id_list.append(int((turn_data['others']['turn'] - 1) / 2))
            # import ipdb; ipdb.set_trace()
            for s_, a_, next_s_ in tqdm(zip(s, a, next_s)):
                if a_.nonzero().shape[0] == 0:
                    continue

                # print([cfg.idx2da[x] for x in a_.nonzero().squeeze(1).tolist()])
                if turn_id_list[cur_id + 1] == 0 or cur_id + 1 == len(turn_id_list):
                    a_next_ = torch.zeros_like(a_)
                else:
                    a_next_ = a[cur_id + 1]

                a_seq_next.append(torch.Tensor(a_next_.nonzero().squeeze(1).tolist()))
                # import ipdb;ipdb.set_trace()
                a_pos_seq.append(torch.Tensor([a_.nonzero().shape[0] + a_next_.nonzero().shape[0]]))
                # if not final_turn:
                # import ipdb; ipdb.set_trace()
                len_now = a_.nonzero().shape[0]
                if len_now not in dist_dict:
                    dist_dict[len_now] = 1
                else:
                    dist_dict[len_now] += 1

                n_s.append(s_)
                n_next_s.append(next_s_)
                s_pos_seq.append(torch.Tensor([a_.nonzero().shape[0]]))
                a_seq.append(torch.Tensor(a_.nonzero().squeeze(1).tolist()))
                cur_id += 1
            dataset = DatasetPEDPnew(n_s, a_seq, n_next_s, s_pos_seq, a_seq_next, a_pos_seq)
        else:
            print(other_data + '{}.pt'.format(part))
            n_s, a_seq, n_next_s, s_pos_seq, a_seq_next, a_pos_seq, _ = torch.load(other_data + '{}.pt'.format(part))
            dataset = DatasetPEDPnew(n_s, a_seq, n_next_s, s_pos_seq, a_seq_next, a_pos_seq)
            print(other_data + '{}.pt  total of {} samples'.format(part, len(dataset)))

        def col_fn(batch):
            """
            prepare batch
            --- for TEXT -> max_len * batch
            - cat stc in the doc
            - sort by doc_text len & store max_len
            - build doc_text order index
            --- for target -> catted 1D longTensor
            """
            b_s, b_a_seq, b_s_next, b_s_pos_seq, b_a_seq_aug = [], [], [], [], []
            b_a_seq_full, b_a_pos_seq = [], []
            for cb_s, cb_a_seq, cb_s_next, cb_s_pos_seq, cb_a_seq_next, cb_a_pos_seq in batch:
                # import ipdb;ipdb.set_trace()
                b_s.append(cb_s)
                cp_cb_a_seq = deepcopy(cb_a_seq.tolist())
                cp_cb_a_seq.sort()
                # cp_cb_a_seq.insert(0, 166)
                # import ipdb;ipdb.set_trace()
                cp_cb_a_seq.append(cfg.a_dim + 1)
                b_a_seq_aug.append(torch.Tensor(pad_to(cfg.max_len,
                                                       cp_cb_a_seq, True, pad_id=cfg.a_dim + 2)))

                cb_a_seq = cb_a_seq.tolist()
                random.shuffle(cb_a_seq)
                b_a_seq.append(torch.Tensor(pad_to(cfg.max_len, cb_a_seq, True)))

                cb_a_seq_next = cb_a_seq_next.tolist()
                random.shuffle(cb_a_seq_next)
                cb_a_seq.extend(cb_a_seq_next)
                b_a_seq_full.append(torch.Tensor(pad_to(cfg.max_len, cb_a_seq, True)))

                b_a_pos_seq.append(cb_a_pos_seq)

                b_s_next.append(cb_s_next)
                b_s_pos_seq.append(cb_s_pos_seq)
            return [torch.stack(b_s), torch.stack(b_a_seq), torch.stack(b_s_next),
                    torch.stack(b_s_pos_seq), torch.stack(b_a_seq_aug), torch.stack(b_a_seq_full), torch.stack(b_a_pos_seq)]

        if part == 'train':
            # dataloader = data.DataLoader(dataset, batchsz, shuffle=True)
            dataloader = data.DataLoader(dataset, batchsz, shuffle=True, collate_fn=col_fn, drop_last=True)
        else:
            dataloader = data.DataLoader(dataset, batchsz, shuffle=False, collate_fn=col_fn, drop_last=True)
        logging.info('MultiWOZ | DataLoader {} | Total Batches {} | Batch Size {} | time {:.3}'.format(part, len(dataloader), batchsz, time.time()-t0))
        logging.info('=' * 100)
        return dataloader
    #
    # def create_dataset_PEDP(self, part, batchsz, cfg, db, full_data=False):
    #     logging.info('=' * 100)
    #     load_file_dir = self.data_dir_new + '/' + part + '.pt'
    #     aug_file_dir = self.data_dir_new + '/' + part + '_aug_wozero_seq.pt'
    #     # aug_file_dir = self.data_dir_new + '/' + part + '_full.pt'
    #     if not os.path.exists(load_file_dir):
    #         logging.info('MultiWOZ | Create Dataset: {}'.format(load_file_dir))
    #         self.create_dataset(part, load_file_dir, cfg, db)
    #     t0 = time.time()
    #     # if not os.path.exists(aug_file_dir):
    #     s, a, next_s, _ = torch.load(load_file_dir)
    #     logging.info('Source Loaded: Total {} samples'.format(len(a)))
    #     n_s, n_next_s, a_seq, s_pos_seq = [], [], [], []
    #     a_next_seq = []
    #     dist_dict = dict()
    #     for s_, a_, next_s_ in zip(s, a, next_s):
    #         len_now = a_.nonzero().shape[0]
    #         if len_now not in dist_dict:
    #             dist_dict[len_now] = 1
    #         else:
    #             dist_dict[len_now] += 1
    #         if not full_data:
    #             if a_.nonzero().shape[0] == 0:
    #                 continue
    #         n_s.append(s_)
    #         n_next_s.append(next_s_)
    #         s_pos_seq.append(torch.Tensor([a_.nonzero().shape[0]]))
    #         a_seq.append(torch.Tensor(a_.nonzero().squeeze(1).tolist()))
    #     dataset = DatasetPEDP(n_s, a_seq, n_next_s, s_pos_seq)
    #         # torch.save(dataset, aug_file_dir)
    #     # else:
    #     #     dataset = torch.load(aug_file_dir)
    #
    #     def col_fn(batch):
    #         """
    #         prepare batch
    #         --- for TEXT -> max_len * batch
    #         - cat stc in the doc
    #         - sort by doc_text len & store max_len
    #         - build doc_text order index
    #         --- for target -> catted 1D longTensor
    #         """
    #         b_s, b_a_seq, b_s_next, b_s_pos_seq, b_a_seq_aug = [], [], [], [], []
    #         b_a_seq_sorted = []
    #         for cb_s, cb_a_seq, cb_s_next, cb_s_pos_seq in batch:
    #             b_s.append(cb_s)
    #             cp_cb_a_seq = deepcopy(cb_a_seq.tolist())
    #             cp_cb_a_seq.sort()
    #             # cp_cb_a_seq.insert(0, 166)
    #             cp_cb_a_seq.append(167)
    #             b_a_seq_aug.append(torch.Tensor(pad_to(cfg.max_len, cp_cb_a_seq, True, pad_id=168)))
    #
    #             cb_a_seq = cb_a_seq.tolist()
    #             random.shuffle(cb_a_seq)
    #
    #             b_a_seq.append(torch.Tensor(pad_to(cfg.max_len, cb_a_seq, True)))
    #
    #             b_s_next.append(cb_s_next)
    #             b_s_pos_seq.append(cb_s_pos_seq)
    #         return [torch.stack(b_s), torch.stack(b_a_seq), torch.stack(b_s_next), torch.stack(b_s_pos_seq), torch.stack(b_a_seq_aug)]
    #
    #     if part == 'train':
    #         # dataloader = data.DataLoader(dataset, batchsz, shuffle=True)
    #         dataloader = data.DataLoader(dataset, batchsz, shuffle=True, collate_fn=col_fn)
    #     else:
    #         dataloader = data.DataLoader(dataset, batchsz, shuffle=False, collate_fn=col_fn)
    #     logging.info('MultiWOZ | DataLoader {} | Total Batches {} | Batch Size {} | time {:.3}'.format(part, len(dataloader), batchsz, time.time()-t0))
    #     logging.info('=' * 100)
    #     return dataloader


    def create_dataset_gcas(self, part, batchsz, cfg, db):
        # logging.info('=' * 100)
        # load_file_dir = self.data_dir_new + '/' + part + '.pt'
        # if not os.path.exists(load_file_dir):
        #     logging.info('MultiWOZ | Create Dataset: {}'.format(load_file_dir))
        #     self.create_dataset(part, load_file_dir, cfg, db)
        t0 = time.time()
        # s, a, next_s, _ = torch.load(load_file_dir)
        s_load, a_load, _, _, _, _, _ = torch.load(self.data_dir[:-4] + '/sgd_data/processed/{}.pt'.format(part))
        logging.info('Source Loaded: Total {} samples'.format(len(a_load)))
        n_s, n_gcas_c, n_gcas_a, n_gcas_s, n_atgt = [], [], [], [], []
        for s_, a_ in tqdm(zip(s_load, a_load)):
            tmp = torch.zeros(cfg.a_dim)
            for a_idx in a_.long().tolist():
                tmp[a_idx] = 1
            a_ = tmp
            if a_.nonzero().shape[0] == 0:
                continue
            n_s.append(s_)
            n_atgt.append(a_)
            a_seq_lst = a_.nonzero().squeeze(1).tolist()
            a_ori_lst = [cfg.idx2da[x] for x in a_seq_lst]

            di_lst, sv_lst, continue_lst = [], [], []
            di2sv = dict()
            for ori_act in a_ori_lst:
                # di_ori = '-'.join(ori_act.split('-')[:2])
                # sv_ori = '-'.join(ori_act.split('-')[2:])
                di_ori = ori_act.split('-')[1]
                sv_ori = ori_act.split('-')[0] + '-' + '-'.join(ori_act.split('-')[2:])
                # import ipdb; ipdb.set_trace()
                di_idx = cfg.di2idx[di_ori]
                sv_idx = cfg.sv2idx[sv_ori]
                if di_idx not in di2sv:
                    di2sv[di_idx] = []
                di2sv[di_idx].append(sv_idx)

            for di_idx_, sv_idx_ in di2sv.items():
                di_lst.append(di_idx_)
                sv_lst.append(sv_idx_)
                continue_lst.append(1)
            continue_lst.append(0)

            n_gcas_c.append(continue_lst)
            n_gcas_a.append(di_lst)
            n_gcas_s.append(sv_lst)

        dataset = DatasetGCAS(n_s, n_gcas_c, n_gcas_a, n_gcas_s, n_atgt)

        def col_fn(batch):
            """
            prepare batch
            --- for TEXT -> max_len * batch
            - cat stc in the doc
            - sort by doc_text len & store max_len
            - build doc_text order index
            --- for target -> catted 1D longTensor
            """
            b_s, b_c, b_a, b_ss, b_a_tgt = [], [], [], [], []
            for cb_s, cb_c, cb_a, cb_ss, cb_a_tgt in batch:
                b_s.append(cb_s)
                b_a_tgt.append(cb_a_tgt)
                # 13, 14 intent
                b_a.append(torch.Tensor(pad_to(cfg.max_len, cb_a, True, pad_id=cfg.gcas_a_dim)))
                # 111, 112, 113 dsv
                # cb_ss.insert(0, 111)
                # b_ss.append(torch.Tensor(pad_to(cfg.max_len, cb_ss, True, pad_id=112)))
                slot_cb = []
                for slot_set in cb_ss:
                    slot_vec = torch.zeros(cfg.gcas_s_dim)
                    for slot_ in slot_set:
                        slot_vec[slot_] = 1
                    slot_cb.append(slot_vec)
                # -- pad and concat
                if len(slot_cb) < cfg.max_len:
                    slot_cb.extend([torch.zeros(cfg.gcas_s_dim)] * (cfg.max_len - len(slot_cb)))
                b_ss.append(torch.stack(slot_cb))

                # 2, 3, 4 continue
                b_c.append(torch.Tensor(pad_to(cfg.max_len, cb_c, True, pad_id=cfg.gcas_c_dim)))

            return [torch.stack(b_s), torch.stack(b_c), torch.stack(b_a), torch.stack(b_ss), torch.stack(b_a_tgt)]

        if part == 'train':
            # dataloader = data.DataLoader(dataset, batchsz, shuffle=True)
            dataloader = data.DataLoader(dataset, batchsz, shuffle=True, collate_fn=col_fn)
        else:
            dataloader = data.DataLoader(dataset, batchsz, shuffle=False, collate_fn=col_fn)
        logging.info('MultiWOZ | DataLoader {} | Total Batches {} | Batch Size {} | time {:.3}'.format(part, len(dataloader), batchsz, time.time()-t0))
        logging.info('=' * 100)
        return dataloader

    def create_dataset_seq_pro(self, part, batchsz, cfg, db, curriculum=0, fetch=False, just_build=False, advance=False, world=False):
        do_aug = False if cfg.aug_type == 'none' else True
        logging.info('=' * 100)
        self.loaded_data = torch.load(self.data_dir_new + '/' + part + '.pt')
        s, a, next_s, term = self.loaded_data
        import ipdb; ipdb.set_trace()
        if do_aug:
            seq_file_dir = self.data_dir_new + '/' + part + '_aug_c{}.pt'.format(cfg.aug_type, curriculum)
        else:
            seq_file_dir = self.data_dir_new + '/' + part + '_c{}.pt'.format(curriculum)
        if not os.path.exists(seq_file_dir) or just_build or advance:
            logging.info('MultiWOZ | Build: {}'.format(seq_file_dir))
            file_dir = self.data_dir_new + '/' + part + '_source.pt'
            if not os.path.exists(file_dir) or just_build:
                logging.info('MultiWOZ | Create Dataset: {}'.format(file_dir))
                self.create_dataset(part, file_dir, cfg, db)
                if just_build:
                    return
            start_time = time.time()
            if part == 'train':
                if self.loaded_data is None:
                    self.loaded_data = torch.load(file_dir)
                s, a, next_s, d, a_seq = self.loaded_data
            else:
                s, a, next_s, d, a_seq = torch.load(file_dir)
            ds_all = len(s)
            logging.info('MultiWOZ | Load Augmented Dataset: {} | Samples: {} | Time Cost: {:.1} s'.format(file_dir, ds_all, time.time()-start_time))
            if part != 'train':
                turn_id_list_val = []
                for i, turn_data in enumerate(self.data[part]):
                    if i % 2 == 1:
                        turn_id_list_val.append(int((turn_data['others']['turn'] - 1) / 2))
                new_a_seq = []
                for act_seq_padded in a_seq:
                    sel_index = torch.arange(1, (act_seq_padded == 2).nonzero(as_tuple=True)[0][-1].item())
                    act_seq = torch.index_select(act_seq_padded, -1, sel_index).tolist()
                    new_a_seq.append(act_seq)
                a_seq = new_a_seq
                new_a_seq = []
                new_s_seq = []
                s_pos_seq = []
                ori_s_seq = []
                for cur_s, cur_id, act_seq, next_s_sample, turn_id_val in zip(s, list(range(len(a_seq))), a_seq, next_s, turn_id_list_val):
                    if len(act_seq) == 0:
                        continue
                    ori_s_seq.append(cur_s)
                    default_seq = act_seq
                    # if cur_id != len(a_seq) - 1:
                    #     if turn_id_val < turn_id_list_val[cur_id + 1]:
                    #         default_seq.extend(a_seq[cur_id + 1])
                    # default_seq.append(170)
                    default_seq = pad_to(20, default_seq, True)
                    tmp_seq = deepcopy(act_seq)
                    tmp_seq = sorted(tmp_seq)
                    s_pos_seq.append(torch.Tensor([len(tmp_seq)]))
                    default_seq.extend(tmp_seq)
                    default_seq.append(2)
                    default_seq = pad_to(40, default_seq, True)
                    new_a_seq.append(torch.Tensor(default_seq))
                    new_s_seq.append(next_s_sample)
                s = ori_s_seq
                a_seq = new_a_seq
                if fetch:
                    return s, a_seq

                dataset = DatasetSeqPro(s, a_seq, new_s_seq, s_pos_seq)
                dataloader = data.DataLoader(dataset, batchsz, shuffle=True, drop_last=True)
                logging.info('MultiWOZ | DataLoader {} | Total Batches {} | Batch Size {} | Shuffle: {}'.format(part,
                                                                                                                len(dataloader),
                                                                                                                batchsz,
                                                                                                                True))
                logging.info('=' * 100)
                return dataloader

            # -- turn id list
            turn_id_list = []
            turn_cnt, session_cnt = 0, 0
            for i, turn_data in enumerate(self.data[part]):
                if i % 2 == 1:
                    cur_id = int((turn_data['others']['turn'] - 1)/2)
                    turn_cnt += 1
                    if cur_id == 0:
                        session_cnt += 1
                    turn_id_list += [cur_id] * len(turn_data['sys_action_aug'])
            logging.info('MultiWOZ | {} | total sessions: {} | total turns: {}'.format(part, session_cnt, turn_cnt))

            # -- un-pad
            new_a_seq = []
            for act_seq_padded in a_seq:
                sel_index = torch.arange(1, (act_seq_padded == 2).nonzero(as_tuple=True)[0][-1].item())
                act_seq = torch.index_select(act_seq_padded, -1, sel_index).tolist()
                new_a_seq.append(act_seq)
            a_seq = new_a_seq
            smp_num_counter = []
            sample_cnt = 0

            # -- store act seq of len curriculum (with curriculum <= session length) for every state
            # -- here 0 is 2 turns >
            for data_curriculum in range(1,2):
                max_act_len, len_dist_dict = 0, dict()
                s_data, a_seq_data, s_seq_data, s_pos_data = [], [], [], []
                last_turn_id = -1

                for start_idx, start_turn_id in tqdm(zip(list(range(len(s))), turn_id_list)):
                    # -- filter out repeating ones
                    if start_turn_id == last_turn_id:
                        smp_num_counter[-1] += 1
                    else:
                        smp_num_counter.append(1)
                    last_turn_id = start_turn_id
                if cfg.aug_type == 'mean':
                    aug_num = int(np.mean(smp_num_counter)) + 1
                elif cfg.aug_type == 'max':
                    aug_num = max(smp_num_counter) // 2
                elif cfg.aug_type == 'base':
                    # -- base
                    aug_num = 1
                else:
                    aug_num = 0

                for start_idx, s_sample, a_seq_sample, start_turn_id, next_s_sample \
                        in tqdm(zip(list(range(len(s))), s, a_seq, turn_id_list, next_s)):
                    # -- filter out repeating ones
                    if start_turn_id == last_turn_id and not do_aug:
                        continue
                    if world:
                        if len(a_seq_sample) != 1:
                            continue
                    last_turn_id = start_turn_id
                    # -- turn_id_list: [0, 0, 1, 2, 2, 2]
                    # -- target_idx_seqs: get next idx seqs like  [[2], [3, 4, 5]] for 0 and 1
                    target_idx_seqs = []
                    end_turn_id = start_turn_id + data_curriculum + 1
                    end_idx = start_idx
                    last_note_turn_id = start_turn_id
                    for next_turn_id in turn_id_list[start_idx + 1:]:
                        # -- current index
                        end_idx += 1
                        # -- break if step into further turns or next session
                        if next_turn_id > end_turn_id or next_turn_id < start_turn_id:
                            break
                        # -- skip repeat items in current turn
                        if next_turn_id == start_turn_id:
                            continue
                        # -- if first step into target turn
                        if next_turn_id > last_note_turn_id:
                            target_idx_seqs.append([end_idx])
                            last_note_turn_id = next_turn_id
                        else:
                            target_idx_seqs[-1].append(end_idx)
                    # -- skip if not enough for curriculum
                    # if len(target_idx_seqs) < data_curriculum:
                    #     continue

                    # -- add data
                    # -- get possible idx paths
                    num_paths = np.prod([len(x) for x in target_idx_seqs])
                    possible_idx_paths = []
                    while len(possible_idx_paths) < num_paths:
                        cur_path_sample = [choice(idx_lst) for idx_lst in target_idx_seqs]
                        if cur_path_sample not in possible_idx_paths:
                            possible_idx_paths.append(cur_path_sample)

                    # -- add data for each path
                    # import ipdb;
                    # ipdb.set_trace()
                    if not do_aug:
                        possible_idx_paths = [choice(possible_idx_paths)]

                    for idx_path_ in possible_idx_paths:
                        cur_a_seq = []
                        cur_a_seq.extend(a_seq_sample)
                        # for act_idx in idx_path_:
                        #     cur_a_seq.extend(a_seq[act_idx])

                        if len(a_seq_sample) * np.prod([len(a_seq[x]) for x in idx_path_]) == 0:
                            continue
                        # cur_a_seq.append(170)
                        cur_a_seq = pad_to(20, cur_a_seq, True)
                        tmp_seq = deepcopy(a_seq_sample)
                        tmp_seq = sorted(tmp_seq)
                        s_pos_data.append(torch.Tensor([len(tmp_seq)]))
                        cur_a_seq.extend(tmp_seq)
                        cur_a_seq.append(2)
                        cur_a_seq = pad_to(40, cur_a_seq, True)
                        s_data.append(s_sample)
                        s_seq_data.append(next_s_sample)
                        a_seq_data.append(torch.Tensor(cur_a_seq))
                        sample_cnt += 1

                    if aug_num > 1:
                        for i in range(aug_num - 1):
                            s_data.append(s_data[-1])
                            a_seq_data.append(a_seq_data[-1])
                            s_seq_data.append(s_seq_data[-1])
                            s_pos_data.append(s_pos_data[-1])
                            sample_cnt += 1

                logging.info('MultiWOZ | Curriculum: {} | Samples: {} | Aug Type: {} | mean_max_aug: {} vs {} | Act Length Distribution: {}'
                             .format(data_curriculum, len(s_data), cfg.aug_type, np.mean(smp_num_counter), max(smp_num_counter), [(k, len_dist_dict[k]) for k in sorted(len_dist_dict.keys())]))
                logging.info('MultiWOZ | Save as: {}'.format(seq_file_dir))
                torch.save((s_data, a_seq_data, s_seq_data, s_pos_data), seq_file_dir)
        s_seq, a_tot_seq, s_tot_seq, s_pos_tot_seq = torch.load(seq_file_dir)
        logging.info('MultiWOZ | Load: {} | Samples: {}'.format(seq_file_dir, len(s_seq)))
        if fetch:
            return s_seq, a_tot_seq, s_tot_seq
        dataset = DatasetSeqPro(s_seq, a_tot_seq, s_tot_seq, s_pos_tot_seq)
        dataloader = data.DataLoader(dataset, batchsz, shuffle=True, drop_last=True)
        logging.info('MultiWOZ | DataLoader {} | Total Batches {} | Batch Size {} | Shuffle: {}'.format(part, len(dataloader), batchsz, True))
        logging.info('=' * 100)
        return dataloader
    

class DatasetPol(data.Dataset):
    def __init__(self, s_s, a_s):
        self.s_s = s_s
        self.a_s = a_s
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        return s, a
    
    def __len__(self):
        return self.num_total


class DatasetGCAS(data.Dataset):
    def __init__(self, s, c, a, ss, n_atgt):
        self.s = s
        self.c = c
        self.a = a
        self.ss = ss
        self.n_atgt = n_atgt
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        c = self.c[index]
        a = self.a[index]
        ss = self.ss[index]
        n_atgt = self.n_atgt[index]
        return s, c, a, ss, n_atgt

    def __len__(self):
        return self.num_total

class DatasetPEDPnew(data.Dataset):
    def __init__(self, s, a_seq, s_next, s_pos_seq, a_seq_next, a_pos_seq):
        self.s = s
        self.a_seq = a_seq
        self.s_next = s_next
        self.s_pos_seq = s_pos_seq
        self.a_seq_next = a_seq_next
        self.a_pos_seq = a_pos_seq
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        a_seq = self.a_seq[index]
        s_next = self.s_next[index]
        s_pos_seq = self.s_pos_seq[index]
        a_pos_seq = self.a_pos_seq[index]
        a_seq_next = self.a_seq_next[index]
        return s, a_seq, s_next, s_pos_seq, a_seq_next, a_pos_seq

    def __len__(self):
        return self.num_total


class DatasetPEDP(data.Dataset):
    def __init__(self, s, a_seq, s_next, s_pos_seq):
        self.s = s
        self.a_seq = a_seq
        self.s_next = s_next
        self.s_pos_seq = s_pos_seq
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        a_seq = self.a_seq[index]
        s_next = self.s_next[index]
        s_pos_seq = self.s_pos_seq[index]
        return s, a_seq, s_next, s_pos_seq

    def __len__(self):
        return self.num_total

class DatasetPEDP_old(data.Dataset):
    def __init__(self, s, a_seq, s_next, term, label, s_pos_seq):
        self.s = s
        self.a_seq = a_seq
        self.s_next = s_next
        self.term = term
        self.label = label
        self.s_pos_seq = s_pos_seq
        self.num_total = len(s)

    def __getitem__(self, index):
        s = self.s[index]
        a_seq = self.a_seq[index]
        s_next = self.s_next[index]
        term = self.term[index]
        label = self.label[index]
        s_pos_seq = self.s_pos_seq[index]
        return s, a_seq, s_next, s_pos_seq, term, label

    def __len__(self):
        return self.num_total


class DatasetPlanner(data.Dataset):
    def __init__(self, s_s, a_s, cur):
        self.s_s = s_s
        self.a_s = a_s
        self.cur = cur
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        cur = self.cur[index]
        return s, a, cur

    def __len__(self):
        return self.num_total


class DatasetWorld(data.Dataset):
    def __init__(self, s_s, a_s, next_s_s):
        self.s_s = s_s
        self.a_s = a_s
        self.next_s_s = next_s_s
        #self.t = t
        self.num_total = len(s_s)
    
    def __getitem__(self, index):
        s = self.s_s[index]
        a = self.a_s[index]
        next_s = self.next_s_s[index]
       # t = self.t[index]
        return s, a, next_s
    
    def __len__(self):
        return self.num_total


class DatasetSeq(data.Dataset):
    def __init__(self, s_s, a_tot_seq, s_tot_seq):
        self.s_s = s_s
        self.a_tot_seq = a_tot_seq
        self.s_tot_seq = s_tot_seq
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a_tot_seq = self.a_tot_seq[index]
        s_tot_seq = self.s_tot_seq[index]
        return s, a_tot_seq, s_tot_seq

    def __len__(self):
        return self.num_total


class DatasetSeqPro(data.Dataset):
    def __init__(self, s_s, a_tot_seq, s_tot_seq, s_pos):
        self.s_s = s_s
        self.a_tot_seq = a_tot_seq
        self.s_tot_seq = s_tot_seq
        self.s_pos = s_pos
        self.num_total = len(s_s)

    def __getitem__(self, index):
        s = self.s_s[index]
        a_tot_seq = self.a_tot_seq[index]
        s_tot_seq = self.s_tot_seq[index]
        s_pos = self.s_pos[index]
        return s, a_tot_seq, s_tot_seq, s_pos

    def __len__(self):
        return self.num_total
