import logging
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
from agenda.agenda import UserAgenda
from utils import state_vectorize, id2onehot
from evaluator.metrics import Evaluator
from nlg.multiwoz_template_nlg import MultiwozTemplateNLG
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AgendaEvaluator:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.env = UserAgenda(args.data_dir, cfg)

        self.evaluator = Evaluator(args.data_dir, cfg)


    def post_process_da(self, goal, is_user, dialog):
        dialog_acts = {}
        for slot, value in dialog.items():
            intent = slot.split('-')

            if is_user and intent[1] == 'inform' and value == 'none':
                if intent[0] in goal:
                    if 'book' in goal[intent[0]]:
                        if intent[2] in goal[intent[0]]['book']:
                            value = goal[intent[0]]['book'][intent[2]]
            if value == 'none' and not intent[2] == 'none':
                value = '#' + intent[0] + '-' + intent[1] + '-' + intent[2] + '#'

            if not intent[0] == 'general':
                intent[0] = intent[0].capitalize()
                intent[1] = intent[1].capitalize()
            domain = intent[0] + '-' + intent[1]

            if not intent[0] == 'general':
                dialog_acts.setdefault(domain, [])
                if not intent[2] == 'none':
                    intent[2] = intent[2].capitalize()
                dialog_acts[domain].append([intent[2], value])

        for slot, value in dialog.items():
            intent = slot.split('-')
            if not intent[0] == 'general':
                intent[0] = intent[0].capitalize()
                intent[1] = intent[1].capitalize()
            domain = intent[0] + '-' + intent[1]
            if intent[0] == 'general':
                dialog_acts.setdefault(domain, [])
                if not intent[2] == 'none':
                    intent[2] = intent[2].capitalize()
                dialog_acts[domain].append([intent[2], value])
        return dialog_acts


    def evaluate(self, policy, discriminator=None, save_dialog=False):
        policy.eval()
        if save_dialog:
            saved_goal_list = torch.load('/data2t/szhangspace/EP4MADP/data/saved_goals.txt')
        collected_dialog = []
        traj_len = 40
        reward_tot, turn_tot, inform_tot, match_tot, success_tot = [], [], [], [], []
        loops = 1000
        iterator_agenda = tqdm(range(loops)) if self.args.test else range(loops)

        dialog_human = []
        for seed in range(loops):
            # -- reset and collect user_action, last_sys_action, ..., user_action
            dialog_list = []

            if save_dialog:
                s = self.env.reset(seed, saved_goal=saved_goal_list[seed])
            else:
                s = self.env.reset(seed, saved_goal=None)

            dialog_list.append(s['user_action'])

            turn, reward, mask = traj_len, [], []
            logging.debug('=' * 100)
            logging.debug(self.env.goal.domain_goals)
            goal = self.env.goal.domain_goals

            current_dialog_human = ['GOAL: {}'.format(goal)]

            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, self.env.cfg, self.env.db)).to(device=DEVICE)
                # s_vec = id2onehot(s_vec).to(DEVICE)
                a = policy.select_action(s_vec)

                next_s, done = self.env.step(s, a.cpu())
                s = next_s
                # -- from disc models
                a_expand = id2onehot(a)
                r = discriminator.estimate(s_vec, a_expand.view(-1)).item() if discriminator else 0
                reward.append(r)

                dialog_list.append(s['last_sys_action'])
                # ----------------------------------------------
                multiwoz_template_nlg = MultiwozTemplateNLG(True, self.args)
                input_user = self.post_process_da(goal, True, dialog_list[-2])
                # logging.debug('USER:    ' + multiwoz_template_nlg.generate(input_user))
                current_dialog_human.append('USER: {}'.format(multiwoz_template_nlg.generate(input_user)))

                # logging.debug(dialog_list[-1])
                input_sys = self.post_process_da(goal, False, dialog_list[-1])
                multiwoz_template_nlg = MultiwozTemplateNLG(False, self.args)
                # logging.debug('SYSTEM:  ' + multiwoz_template_nlg.generate(input_sys))
                current_dialog_human.append('SYSTEM: {}'.format(multiwoz_template_nlg.generate(input_sys)))

                if self.args.debug:
                    import ipdb;
                    ipdb.set_trace()

                dialog_list.append(s['user_action'])

                if done:
                    mask.append(0)
                    turn = t + 2  # -- one due to counting from 0, the one for the last turn
                    break
                mask.append(1)
            dialog_human.append(current_dialog_human)
            reward_tot.append(np.mean(reward))
            turn_tot.append(turn)
            match_tot += self.evaluator.match_rate(s)
            inform_tot.append(self.evaluator.inform_F1(s))
            match_session = self.evaluator.match_rate(s, True)
            inform_session = self.evaluator.inform_F1(s, True)
            logging.debug(' ************* OVERALL *************')
            logging.debug("turn: {} | match: {} | inform {}".format(turn, match_session, inform_session))
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                success_tot.append(1)
                logging.debug('success')
            else:
                success_tot.append(0)
                logging.debug('fail')

            dialog_dict = {
                'goal id': seed,
                'goal': self.env.goal.domain_goals,
                'dialog': dialog_list,
                'turn': turn,
                'status': success_tot[-1]
            }

            collected_dialog.append(dialog_dict)

        logging.info('====== robust evaluation on policy ======')
        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))

        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))
        # import ipdb; ipdb.set_trace()
        policy.train()
        if save_dialog:
            if not os.path.exists(self.args.dialog_save_dir):
                os.makedirs(self.args.dialog_save_dir)
            des_path = self.args.dialog_save_dir + '/' + 'collected_dialog.json'
            with open(des_path, 'w') as f:
                json.dump(collected_dialog, f, indent=4)

        return rec, F1, np.mean(success_tot), np.mean(reward_tot), np.mean(turn_tot), np.mean(match_tot)
