# -*- coding: utf-8 -*-
"""
/////
"""
import time
import logging
import os
import operator
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.module import _addindent
from queue import PriorityQueue
from collections import defaultdict
import random

INT = 0
LONG = 1
FLOAT = 2
EOS = 2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=6666):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_session(key, cfg):
    turn_data = {}
    turn_data['others'] = {'session_id': key, 'turn': 0, 'terminal': False}
    turn_data['sys_action'] = dict()
    turn_data['user_action'] = dict()
    turn_data['history'] = {'sys': dict(), 'user': dict()}
    turn_data['belief_state'] = {'inform': {}, 'request': {}, 'booked': {}}
    for domain in cfg.belief_domains:
        turn_data['belief_state']['inform'][domain] = dict()
        turn_data['belief_state']['request'][domain] = set()
        turn_data['belief_state']['booked'][domain] = ''

    session_data = {'inform': {}, 'request': {}, 'book': {}}
    for domain in cfg.belief_domains:
        session_data['inform'][domain] = dict()
        session_data['request'][domain] = set()
        session_data['book'][domain] = False

    return turn_data, session_data


def init_goal(dic, goal, cfg):
    for domain in cfg.belief_domains:
        if domain in goal and goal[domain]:
            domain_data = goal[domain]
            # constraint
            if 'info' in domain_data and domain_data['info']:
                for slot, value in domain_data['info'].items():
                    slot = cfg.map_inverse[domain][slot]
                    # single slot value for user goal
                    inform_da = domain + '-' + slot + '-1'
                    if inform_da in cfg.inform_da:
                        dic['inform'][domain][slot] = value
            # booking
            if 'book' in domain_data and domain_data['book']:
                dic['book'][domain] = True
            # request
            if 'reqt' in domain_data and domain_data['reqt']:
                for slot in domain_data['reqt']:
                    slot = cfg.map_inverse[domain][slot]
                    request_da = domain + '-' + slot
                    if request_da in cfg.request_da:
                        dic['request'][domain].add(slot)


def init_logging_handler(log_dir, extra='', level='debug'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    stderr_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('{}/log_{}.txt'.format(log_dir, current_time + extra))
    logging.basicConfig(handlers=[stderr_handler, file_handler])
    logger = logging.getLogger()
    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif level == 'info':
        logger.setLevel(logging.INFO)


def cur_time():
    return time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime())


def to_device(data):
    if type(data) == dict:
        for k, v in data.items():
            data[k] = v.to(device=DEVICE)
    else:
        for idx, item in enumerate(data):
            data[idx] = item.to(device=DEVICE)
    return data


def state_vectorize(state, config, db, noisy=False):
    """
    state: dict_keys(['user_action', 'sys_action', 'user_goal', 'belief_state', 'history', 'others'])
    state_vec: [user_act, last_sys_act, inform, request, book, degree]
    """
    user_act = np.zeros(len(config.da_usr))
    for da in state['user_action']:
        user_act[config.dau2idx[da]] = 1.

    last_sys_act = np.zeros(len(config.da))
    for da in state['last_sys_action']:
        last_sys_act[config.da2idx[da]] = 1.

    user_history = np.zeros(len(config.da_usr))
    for da in state['history']['user']:
        user_history[config.dau2idx[da]] = 1.

    sys_history = np.zeros(len(config.da))
    for da in state['history']['sys']:
        sys_history[config.da2idx[da]] = 1.

    inform = np.zeros(len(config.inform_da))
    for domain in state['belief_state']['inform']:
        for slot, value in state['belief_state']['inform'][domain].items():
            dom_slot, p = domain + '-' + slot + '-', 1
            key = dom_slot + str(p)
            while inform[config.inform2idx[key]]:
                p += 1
                key = dom_slot + str(p)
                if key not in config.inform2idx:
                    break
            else:
                inform[config.inform2idx[key]] = 1.

    request = np.zeros(len(config.request_da))
    for domain in state['belief_state']['request']:
        for slot in state['belief_state']['request'][domain]:
            request[config.request2idx[domain + '-' + slot]] = 1.

    book = np.zeros(len(config.belief_domains))
    for domain in state['belief_state']['booked']:
        if state['belief_state']['booked'][domain]:
            book[config.domain2idx[domain]] = 1.

    degree = db.pointer(state['belief_state']['inform'], config.mapping, config.db_domains, noisy)

    final = 1. if state['others']['terminal'] else 0.

    state_vec = np.r_[user_act, last_sys_act, user_history, sys_history, inform, request, book, degree, final]
    assert len(state_vec) == config.s_dim
    return state_vec


def action_vectorize(action, config, provided_act=None):
    act_vec = np.zeros(config.a_dim)
    used_act = provided_act if provided_act is not None else action['sys_action']
    for da in used_act:
        act_vec[config.da2idx[da]] = 1
    # if act_vec.sum()==0:
    #     da = 'general-unk'
    #     act_vec[config.da2idx[da]] = 1
    return act_vec


def action_seq(action, config, idx_only=False, provided_act=None):
    act_seq = []
    used_act = provided_act if provided_act is not None else action['sys_action']
    for da in used_act:
        act_seq.append(config.da2idx[da])
    if idx_only:
        return act_seq
    # if len(act_seq)==0:
    #     da = 'general-unk'
    #     act_seq.append(config.da2idx[da])

    # act_seq = sorted(act_seq)  # -- !!!!!!!!


    act_seq.insert(0, 1)  # SOS
    act_seq.append(2)  # EOS
    act_seq = pad_to(config.max_len, act_seq, True)
    return act_seq


def count_act(act_seq, config):
    act_cout = defaultdict(int)
    for a_line in act_seq:
        for a in a_line:
            act_name = config.idx2da[int(a.item())]
            act_cout[act_name] += 1
    logging.info(act_cout)
    logging.info(sorted(act_cout, key=act_cout.get, reverse=True))


def pad_to(max_len, tokens, do_pad=True, pad_id=0):
    if len(tokens) >= max_len:
        return tokens[0:max_len - 1] + [tokens[-1]]
    elif do_pad:
        return tokens + [pad_id] * (max_len - len(tokens))
    else:
        return tokens


def domain_vectorize(state, config):
    domain_vec = np.zeros(config.domain_dim)
    for da in state['user_action']:
        domain = da.strip().split('-')[0]
        domain_vec[config.domain_index[domain]] = 1
    return domain_vec


def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return eps.mul(std) + mu

def id2onehot(id_list):
    one_hot = []
    sp = id_list.shape
    a_dim = sp[-1]
    if type(id_list)==torch.Tensor:
        id_list = id_list.view(-1).tolist()
    for id in id_list:
        if id==0:
            one_hot += [1,0]
        elif id==1:
            one_hot += [0,1]
        else:
            raise ValueError("id can only be 0 or 1, but got {}".format(id))
    return torch.FloatTensor(one_hot).view(-1, a_dim * 2).to(DEVICE)

def onehot2id(onehot_list):
    id_list = []
    bs, a_dim = onehot_list.shape
    newlist = onehot_list.view(-1)
    for i in range(0, len(newlist), 2):
        if newlist[i]>= newlist[i+1]:
            id_list.append(0)
        else:
            id_list.append(1)
    return torch.FloatTensor(id_list).view(bs, a_dim//2).to(DEVICE)



class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in self.items():
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack


def summary(model, show_weights=True, show_parameters=True):
    """
    Summarizes torch models by showing trainable parameters and weights.
    """
    tmpstr = model.__class__.__name__ + ' (\n'
    total_params = 0
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params
        # and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = summary(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])
        total_params += params

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ') Total Parameters={}'.format(total_params)
    return tmpstr


def reparameterized_sample(mean, std):
    """using std to sample"""
    eps = torch.FloatTensor(std.size()).normal_().to(DEVICE)
    eps = Variable(eps).to(DEVICE)
    return eps.mul(std).add_(mean).to(DEVICE)


def kld_gauss(mean_1, std_1, mean_2, std_2, valid_pos_tsr):
    """Using std to compute KLD"""
    kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1)
                   + (std_1.pow(2) + (mean_1 - mean_2).pow(2))/std_2.pow(2)
                   - 1) / 2
    return torch.sum(valid_pos_tsr * kld_element) / kld_element.shape[0]


class GumbelConnector(nn.Module):
    def __init__(self, use_gpu):
        super(GumbelConnector, self).__init__()
        self.use_gpu = use_gpu

    def sample_gumbel(self, logits, use_gpu, eps=1e-20):
        u = torch.rand(logits.size())
        sample = Variable(-torch.log(-torch.log(u + eps) + eps))
        sample = cast_type(sample, FLOAT, use_gpu)
        return sample

    def gumbel_softmax_sample(self, logits, temperature, use_gpu):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        eps = self.sample_gumbel(logits, use_gpu)
        y = logits + eps.to(DEVICE)
        return F.softmax(y / temperature, dim=y.dim() - 1)

    def soft_argmax(self, logits, temperature, use_gpu):
        return F.softmax(logits / temperature, dim=logits.dim() - 1)

    def forward(self, logits, temperature=1.0, hard=False,
                return_max_id=False):
        """
        :param logits: [batch_size, n_class] unnormalized log-prob
        :param temperature: non-negative scalar
        :param hard: if True take argmax
        :param return_max_id
        :return: [batch_size, n_class] sample from gumbel softmax
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        _, y_hard = torch.max(y, dim=1, keepdim=True)
        if hard:
            y_onehot = cast_type(Variable(torch.zeros(y.size())), FLOAT, self.use_gpu)
            y_onehot.scatter_(1, y_hard, 1.0)
            y = y_onehot
        if return_max_id:
            return y, y_hard
        else:
            return y

    def forward_ST(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.soft_argmax(logits, temperature, self.use_gpu)
        # y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y

    def forward_ST_gumbel(self, logits, temperature=0.8):
        """
        input: [*, n_class]
        return: [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, self.use_gpu)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y


def cast_type(var, dtype, use_gpu):
    if use_gpu:
        if dtype == INT:
            var = var.type(torch.cuda.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.cuda.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.cuda.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    else:
        if dtype == INT:
            var = var.type(torch.IntTensor)
        elif dtype == LONG:
            var = var.type(torch.LongTensor)
        elif dtype == FLOAT:
            var = var.type(torch.FloatTensor)
        else:
            raise ValueError("Unknown dtype")
    return var


class TrainProgressChecker:
    def __init__(self, patience=3, delta=0, banned=False):
        """
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.banned = banned
        self.last_score = None

    def __call__(self, train_loss):
        """
        loss small = score big
        :param train_loss:
        :return: if making progress
        """
        if self.banned:
            return True
        score = -train_loss
        if self.last_score is not None and self.last_score - score > 0.5:
            self.best_score = score
            self.counter = 0
            return True
        self.last_score = score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return False
        else:
            self.best_score = score
            self.counter = 0
        return True


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward



def beam_decode(decoder, head, emb, decoder_hidden, sos_, eos_, h_0):
    '''
    :param decoder_hidden: input tensor of shape [H] for start of the decoding
    :return: decoded_batch
    '''

    beam_width = 4
    topk = 1  # how many sentence do you want to generate

    # Start with the start of the sentence token
    with torch.no_grad():
        decoder_input = Variable(torch.LongTensor([sos_])).to(DEVICE)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
    nodes = PriorityQueue()
    id_counter = 0
    # start the queue
    nodes.put((-node.eval(), id_counter, node))
    id_counter += 1

    qsize = 1
    tot_step = 1
    # start beam search
    while True:
        # give up when decoding takes too long
        if qsize > 2000: break

        # fetch the best node
        score, _, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h
        tot_step += 1
        if (n.wordid.item() == eos_ and n.prevNode != None) or tot_step == 20:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue
        # decode for one step using decoder
        decoder_hidden = decoder(torch.cat((emb(decoder_input), h_0), dim=-1).unsqueeze(0),
                                 decoder_hidden.unsqueeze(0))[1].squeeze(0)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(F.log_softmax(head(decoder_hidden), -1), beam_width)
        nextnodes = []
        # import ipdb; ipdb.set_trace()
        for new_k in range(beam_width):
            log_p = log_prob[0][new_k].item()
            node = BeamSearchNode(decoder_hidden, n, indexes[0][new_k].view(1), n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]

            nodes.put((score, id_counter, nn))
            id_counter += 1

            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    utterance = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance.append(n.wordid)
        # back trace
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)
        utterance = utterance[::-1]

    last_id = None
    filtered_uttr = []
    for id_tsr in utterance:
        if id_tsr == last_id:
            continue
        filtered_uttr.append(id_tsr)
        last_id = id_tsr

    return torch.cat([emb(x) for x in filtered_uttr], dim=0), [x.unsqueeze(1) for x in filtered_uttr]

