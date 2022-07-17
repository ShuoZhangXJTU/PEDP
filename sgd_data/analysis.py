import os
import json
from copy import deepcopy
from ref import *
import torch
from encoder import *
from config import *
from tqdm import tqdm


class Analyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.all_files = dict()
        self.find_files()

    def find_files(self):
        folders = os.listdir(self.base_path)
        for folder in folders:
            if folder in ['train', 'dev', 'test']:
                self.all_files[folder] = [self.base_path + "/" + folder + "/" + x
                                          for x in os.listdir(self.base_path + "/" + folder)]

    def init_turn_data(self, key):
        turn_data = {}
        turn_data['others'] = {'session_id': key, 'turn': 0, 'terminal': False}
        turn_data['sys_action'] = dict()
        turn_data['user_action'] = dict()
        turn_data['history'] = {'sys': dict(), 'user': dict()}
        turn_data['belief_state'] = {'INFORM': {}, 'REQUEST': {}}
        for domain in DOMAINS:
            turn_data['belief_state']['INFORM'][domain] = dict()
            turn_data['belief_state']['REQUEST'][domain] = set()
        return turn_data

    def prepare_sessions(self, part):
        intents = set()
        valid_num = 0
        all_num = 0
        valid_dirs = {}
        not_in_actions = {
            'USER': set(),
            'SYSTEM': set()
        }
        if not os.path.exists('./processed/{}_raw'.format(part)):
            for file_dir in tqdm(self.all_files[part]):
                if 'schema' in file_dir: continue
                with open(file_dir, 'r', encoding='UTF-8') as f:
                    for k_sess, session in enumerate(json.loads(f.read())):
                        all_num += 1
                        turn_data_skip = False
                        for i, turn in enumerate(session['turns']):
                            for frame in turn['frames']:
                                domain = frame['service'].split('_')[0]
                                for action_dict in frame['actions']:
                                    slot = action_dict['slot']
                                    if domain in COREF:
                                        if slot in COREF[domain]:
                                            slot = COREF[domain][slot]
                                    intent = action_dict['act']
                                    intents.add(intent)
                                    DIS_HERE = domain + '-' + intent + '-' + slot
                                    if domain not in DOMAINS or intent not in INTENTS or slot not in SLOTS[domain]:
                                        turn_data_skip = True
                                        break
                                    if turn['speaker'] == 'USER':
                                        if DIS_HERE not in DAU_FILTER[domain]:
                                            turn_data_skip = True
                                    else:
                                        if DIS_HERE not in DA_FILTER[domain]:
                                            turn_data_skip = True
                                    if turn_data_skip:
                                        break
                                if turn_data_skip:
                                    break
                            if turn_data_skip:
                                break
                        valid_dirs[session['dialogue_id']] = turn_data_skip
                        if turn_data_skip:
                            not_in_actions[turn['speaker']].add(domain + '-' + intent + '-' + slot)
                            continue
                        valid_num += 1
            print(not_in_actions)
            print('valid {} of {}'.format(valid_num, all_num))
            # import ipdb; ipdb.set_trace()

            # -- ref.py
            intents, req_da, inf_da, da, dau = set(), dict(), dict(), dict(), dict()
            slots = dict()

            # -- config.py
            domains = set()

            raw_data = []
            valid_cnt = 0
            for file_dir in tqdm(self.all_files[part]):
                if 'schema' in file_dir: continue
                with open(file_dir, 'r', encoding='UTF-8') as f:
                    for k_sess, session in enumerate(json.loads(f.read())):
                        if valid_dirs[session['dialogue_id']]:
                            continue
                        valid_cnt += 1
                        turn_data = self.init_turn_data(k_sess)
                        belief_state = turn_data['belief_state']
                        for i, turn in enumerate(session['turns']):
                            turn_data['others']['turn'] = i
                            turn_data['others']['terminal'] = i + 2 >= len(session['turns'])
                            turn_data['belief_state'] = deepcopy(belief_state)
                            turn_data['role'] = turn['speaker']
                            turn_data['uttr'] = turn['utterance']

                            if turn['speaker'] == 'USER':
                                if 'last_sys_action' in turn_data:
                                    turn_data['history']['sys'] = dict(turn_data['history']['sys'],
                                                                       **turn_data['last_sys_action'])
                                    del (turn_data['last_sys_action'])
                                turn_data['last_user_action'] = deepcopy(turn_data['user_action'])
                                turn_data['user_action'] = dict()

                                for frame in turn['frames']:
                                    domain = frame['service'].split('_')[0]
                                    domains.add(domain)
                                    for action_dict in frame['actions']:
                                        slot = action_dict['slot']
                                        if domain in COREF:
                                            if slot in COREF[domain]:
                                                slot = COREF[domain][slot]
                                        intent = action_dict['act']
                                        intents.add(intent)

                                        DIS = '-'.join([domain, intent, slot])

                                        if domain not in slots:
                                            slots[domain] = set()
                                        slots[domain].add(slot)

                                        if domain not in dau:
                                            dau[domain] = set()
                                        dau[domain].add(DIS)

                                        turn_data['user_action'][DIS] = True

                                        if 'INFORM' in intent:
                                            belief_state['INFORM'][domain][slot] = True

                                            if domain not in inf_da:
                                                inf_da[domain] = set()
                                            inf_da[domain].add(domain + '-' + slot)

                                        elif 'REQUEST' in intent:
                                            belief_state['REQUEST'][domain].add(slot)

                                            if domain not in req_da:
                                                req_da[domain] = set()
                                            req_da[domain].add(domain + '-' + slot)
                            else:  # sys
                                if 'last_user_action' in turn_data:
                                    turn_data['history']['user'] = dict(turn_data['history']['user'],
                                                                        **turn_data['last_user_action'])
                                    del (turn_data['last_user_action'])
                                turn_data['last_sys_action'] = deepcopy(turn_data['sys_action'])
                                turn_data['sys_action'] = dict()
                                for frame in turn['frames']:
                                    domain = frame['service'].split('_')[0]
                                    domains.add(domain)

                                    for action_dict in frame['actions']:
                                        slot = action_dict['slot']
                                        if domain in COREF:
                                            if slot in COREF[domain]:
                                                slot = COREF[domain][slot]
                                        intent = action_dict['act']
                                        intents.add(intent)

                                        if domain not in slots:
                                            slots[domain] = set()
                                        slots[domain].add(slot)

                                        DIS = '-'.join([domain, intent, slot])

                                        if domain not in da:
                                            da[domain] = set()
                                        da[domain].add(DIS)

                                        turn_data['sys_action'][DIS] = True

                                        if 'INFORM' in intent:
                                            if domain not in inf_da:
                                                inf_da[domain] = set()
                                            inf_da[domain].add(domain + '-' + slot)
                                        elif 'REQUEST' in intent:
                                            if domain not in req_da:
                                                req_da[domain] = set()
                                            req_da[domain].add(domain + '-' + slot)

                                        if 'INFORM' in intent and domain in belief_state['REQUEST']:
                                            belief_state['REQUEST'][domain].discard(slot)

                            if i + 2 >= len(session):
                                turn_data['next_belief_state'] = belief_state
                            raw_data.append(deepcopy(turn_data))
            # import ipdb; ipdb.set_trace()
            torch.save(raw_data, './processed/{}_raw'.format(part))
        else:
            raw_data = torch.load('./processed/{}_raw'.format(part))
            # # CODES FOR SCHEMA CHECK
            # # -- ref.py
            # intents, da, dau = set(), dict(), dict()
            # slots = dict()
            # # -- config.py
            # domains = set()
            #
            # # because state and history may contain, so best to filter in dialog level
            # for turn_data in raw_data:
            #     if turn_data['role'] == 'USER':
            #         action_lst = list(turn_data['user_action'].keys())
            #         for act in action_lst:
            #             domain, intent, slot = act.split('-')
            #             if domain not in dau:
            #                 dau[domain] = set()
            #             dau[domain].add(act)
            #             intents.add(intent)
            #             if domain not in slots:
            #                 slots[domain] = set()
            #             slots[domain].add(slot)
            #             domains.add(domain)
            #     else:
            #         action_lst = list(turn_data['sys_action'].keys())
            #         for act in action_lst:
            #             domain, intent, slot = act.split('-')
            #             if domain not in da:
            #                 da[domain] = set()
            #             da[domain].add(act)
            #             intents.add(intent)
            #             if domain not in slots:
            #                 slots[domain] = set()
            #             slots[domain].add(slot)
            #             domains.add(domain)
            # print(domains)
            # print('---')
            # print(intents)
            # print('---')
            # print(slots)
            # print('---')
            # print(da)
            # print('---')
            # print(dau)
            # import ipdb;ipdb.set_trace()
            print('loaded ./processed/{}_raw'.format(part))
        # print(valid_cnt)
        self.cfg = SGDConfig()
        s, a, next_s, a_seq, next_a, state = [], [], [], [], [], []
        turn_id, term = [], []
        counter = 0
        for idx, turn_data in tqdm(enumerate(raw_data)):
            if turn_data['role'] == 'USER':
                continue

            counter += 1
            print('========== No {} =========='.format(counter))
            for key_, val_ in turn_data.items():
                print('{}: {}'.format(key_, val_))
            aug_action = None
            state_tsr = torch.Tensor(state_vectorize(turn_data, self.cfg))
            s.append(state_tsr)
            act_tsr = torch.Tensor(action_vectorize(turn_data, self.cfg, provided_act=aug_action))
            a.append(act_tsr)
            a_seq.append(torch.Tensor(action_seq(turn_data, self.cfg, provided_act=aug_action)))
            turn_id.append(turn_data['others']['turn'])
            term.append(turn_data['others']['terminal'])
            state_str = str(turn_data['uttr'])
            state.append(state_str)

            if not turn_data['others']['terminal']:
                for to_exam in raw_data[idx + 1:]:
                    if to_exam['role'] == 'SYSTEM':
                        next_s.append(torch.Tensor(state_vectorize(to_exam, self.cfg)))
                        next_a.append(torch.Tensor(action_vectorize(to_exam, self.cfg, provided_act=aug_action)))
                        break
            else:
                next_turn_data = deepcopy(turn_data)
                next_turn_data['others']['turn'] = -1
                next_turn_data['user_action'] = {}
                next_turn_data['last_sys_action'] = next_turn_data['sys_action']
                next_turn_data['sys_action'] = {}
                next_turn_data['belief_state'] = next_turn_data['next_belief_state']
                next_s.append(torch.Tensor(state_vectorize(next_turn_data, self.cfg)))
                next_a.append(torch.zeros_like(act_tsr))

        n_s, n_next_s, a_seq, s_pos_seq, n_state = [], [], [], [], []
        a_seq_next, a_pos_seq = [], []
        dist_dict = dict()
        for s_, a_, next_s_, a_next_, state_ in zip(s, a, next_s, next_a, state):
            len_now = a_.nonzero().shape[0]
            if len_now not in dist_dict:
                dist_dict[len_now] = 1
            else:
                dist_dict[len_now] += 1

            a_seq_next.append(torch.Tensor(a_next_.nonzero().squeeze(1).tolist()))
            a_pos_seq.append(torch.Tensor([a_.nonzero().shape[0] + a_next_.nonzero().shape[0]]))
            n_s.append(s_)
            n_state.append(state_)
            n_next_s.append(next_s_)
            s_pos_seq.append(torch.Tensor([a_.nonzero().shape[0]]))
            a_seq.append(torch.Tensor(a_.nonzero().squeeze(1).tolist()))

        print("----*------")
        torch.save((n_s, a_seq, n_next_s, s_pos_seq, a_seq_next, a_pos_seq, n_state),
                   './processed/{}.pt'.format(part))
        print("----*------")


if __name__ == '__main__':
    analyzer = Analyzer('../sgd_data')
    train_sess = analyzer.prepare_sessions('train')
    dev_sess = analyzer.prepare_sessions('dev')
    valid_sess = analyzer.prepare_sessions('test')
    # with open('./value_dict.json', 'w', encoding='utf-8') as f:
    #     f.write(json.dumps(val_dict_b, ensure_ascii=False))
