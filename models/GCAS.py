import torch
import torch.nn as nn
import random
from torch.autograd import Variable
from utils import GumbelConnector, onehot2id, id2onehot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class gCAS(nn.Module):
    def __init__(self, args, cfg):
        super(gCAS, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_gpu = torch.cuda.is_available()
        self.dropout = nn.Dropout(p=args.dropout)
        self.a_dim = cfg.a_dim
        self.decoder_hidden = cfg.h_dim // 2
        self.head = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                  nn.Tanh(),
                                  nn.Linear(cfg.h_dim, cfg.h_dim // 2))

        self.c_rnn = nn.GRU(self.decoder_hidden, self.decoder_hidden, 1, dropout=args.dropout, bidirectional=False)
        self.a_rnn = nn.GRU(self.decoder_hidden, self.decoder_hidden, 1, dropout=args.dropout, bidirectional=False)
        self.s_rnn = nn.GRU(self.decoder_hidden, self.decoder_hidden, 1, dropout=args.dropout, bidirectional=False)

        self.c_pred = nn.Linear(self.decoder_hidden, cfg.gcas_c_dim + 2)
        self.a_pred = nn.Linear(self.decoder_hidden, cfg.gcas_a_dim + 2)
        self.s_pred = nn.Linear(self.decoder_hidden, cfg.gcas_s_dim)

        self.c_input = torch.nn.Linear(cfg.gcas_c_dim + 2 + cfg.gcas_a_dim + 2 + cfg.gcas_s_dim, self.decoder_hidden)
        self.a_input = torch.nn.Linear(cfg.gcas_c_dim + 2 + cfg.gcas_a_dim + 2 + cfg.gcas_s_dim,
                                       self.decoder_hidden)
        self.s_input = torch.nn.Linear(cfg.gcas_c_dim + 2 + cfg.gcas_a_dim + 2 + cfg.gcas_s_dim,
                                       self.decoder_hidden)
        self.activation = torch.nn.Tanh()
        self.c_loss = nn.CrossEntropyLoss(ignore_index=cfg.gcas_c_dim)
        self.a_loss = nn.CrossEntropyLoss(ignore_index=cfg.gcas_a_dim)
        self.s_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.oov_id = cfg.a_dim


    def select_action(self, s):
        pred_act_tsr = torch.zeros(1, self.a_dim).to(DEVICE)
        s = s.unsqueeze(0)
        h_s = self.head(s)

        # -- predicting
        with torch.no_grad():
            c_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_c_dim + 2)).to(DEVICE)
            # SOS token
            c_sample[:, -1] = 1
            a_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_a_dim + 2)).to(DEVICE)
            a_sample[:, -1] = 1
            s_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_s_dim)).to(DEVICE)

        for step in range(self.cfg.max_len):
            c_input = self.c_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            c_input = self.activation(c_input)
            h_s = self.c_rnn(c_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
            c_weights = self.c_pred(h_s)
            c_sample_num = torch.argmax(c_weights, dim=-1).long()
            c_sample = torch.zeros(s.shape[0], self.cfg.gcas_c_dim + 2).to(DEVICE)
            src_tsr = torch.ones_like(c_sample_num).float().to(DEVICE)
            c_sample.scatter_(-1, c_sample_num.unsqueeze(1), src_tsr.unsqueeze(1))  # -- dim, index, val

            if c_sample_num != 1:
                break

            a_input = self.a_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            a_input = self.activation(a_input)
            h_s = self.a_rnn(a_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
            a_weights = self.a_pred(h_s)
            a_sample_num = torch.argmax(a_weights, dim=-1).long()
            a_sample = torch.zeros(s.shape[0], self.cfg.gcas_a_dim + 2).to(DEVICE)
            src_tsr = torch.ones_like(a_sample_num).float().to(DEVICE)
            a_sample.scatter_(-1, a_sample_num.unsqueeze(1), src_tsr.unsqueeze(1))  # -- dim, index, val

            s_input = self.s_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            s_input = self.activation(s_input)
            h_s = self.s_rnn(s_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze(0)
            s_weights = self.s_pred(h_s)
            s_sample = torch.sigmoid(s_weights).gt(0.5).long()
            pred_act_tsr += self.cas2da(pred_act_tsr, torch.argmax(a_weights, dim=-1).long(),
                                        torch.sigmoid(s_weights).gt(0.5).long(), torch.argmax(c_weights, dim=-1).long())
            pred_act_tsr = pred_act_tsr.ge(1).float()

        return pred_act_tsr


    def cas2da(self, pred_act_tsr, a_sample, s_sample, t_sample):
        pred_act_tsr_ = torch.zeros_like(pred_act_tsr).to(DEVICE)
        for i, a, t in zip(range(a_sample.shape[0]), a_sample.tolist(), t_sample.tolist()):
            if t != 1:
                continue
            if a in self.cfg.idx2di:
                di_str = self.cfg.idx2di[a]
            else:
                continue

            for s in s_sample[i,:].nonzero():
                s = s.item()
                if s in self.cfg.idx2sv:
                    sv_str = self.cfg.idx2sv[s]
                    da = sv_str.split('-')
                    da.insert(1, di_str)
                    da = '-'.join(da)
                    if da in self.cfg.da2idx:
                        pred_act_tsr_[i][self.cfg.da2idx[da]] = 1

        return pred_act_tsr_


    def forward(self, s, c_target_gold, a_target_gold, s_target_gold, train_type='train'):
        # import ipdb; ipdb.set_trace()
        pred_act_tsr = torch.zeros(s.shape[0], self.a_dim).to(DEVICE)
        teacher_forcing = (random.uniform(0, 1) < 0.5) if train_type == 'train' else False
        c_weight_lst = []
        a_weight_lst = []
        s_weight_lst = []
        s_loss = 0.
        loss_num = 0

        h_s = self.head(s)

        # -- predicting
        with torch.no_grad():
            c_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_c_dim + 2)).to(DEVICE)
            c_sample[:, -1] = 1
            a_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_a_dim + 2)).to(DEVICE)
            a_sample[:, -1] = 1
            s_sample = Variable(torch.zeros(s.shape[0], self.cfg.gcas_s_dim)).to(DEVICE)


        for step in range(self.cfg.max_len):
            c_input = self.c_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            c_input = self.activation(c_input)
            h_s = self.c_rnn(c_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze()
            c_weights = self.c_pred(h_s)
            c_weight_lst.append(c_weights.unsqueeze(1))
            if teacher_forcing:
                c_sample_num = c_target_gold[:, step].long()
            else:
                c_sample_num = torch.argmax(c_weights, dim=-1).long()
            c_sample = torch.zeros(s.shape[0], self.cfg.gcas_c_dim + 2).to(DEVICE)
            src_tsr = torch.ones_like(c_sample_num).float().to(DEVICE)
            c_sample.scatter_(-1, c_sample_num.unsqueeze(1), src_tsr.unsqueeze(1))  # -- dim, index, val

            a_input = self.a_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            a_input = self.activation(a_input)
            h_s = self.a_rnn(a_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze()
            a_weights = self.a_pred(h_s)
            a_weight_lst.append(a_weights.unsqueeze(1))
            if teacher_forcing:
                a_sample_num = a_target_gold[:, step].long()
            else:
                a_sample_num = torch.argmax(a_weights, dim=-1).long()
            a_sample = torch.zeros(s.shape[0], self.cfg.gcas_a_dim + 2).to(DEVICE)
            src_tsr = torch.ones_like(a_sample_num).float().to(DEVICE)
            a_sample.scatter_(-1, a_sample_num.unsqueeze(1), src_tsr.unsqueeze(1))  # -- dim, index, val

            s_input = self.s_input(torch.cat((c_sample, a_sample, s_sample), dim=-1))
            s_input = self.activation(s_input)
            h_s = self.s_rnn(s_input.unsqueeze(0), h_s.unsqueeze(0))[1].squeeze()
            s_weights = self.s_pred(h_s)
            s_weight_lst.append(s_weights.unsqueeze(1))
            if teacher_forcing:
                s_sample = s_target_gold[:, step].long()
            else:
                s_sample = torch.sigmoid(s_weights).gt(0.5).long()

            if torch.sum((c_target_gold[:, step] == 1).float()) != 0:
                loss_num += torch.sum((c_target_gold[:, step] == 1).float())
                s_loss += torch.sum((c_target_gold[:, step] == 1).float().unsqueeze(1) * self.s_loss(s_weights, s_target_gold[:, step, :]))

            pred_act_tsr += self.cas2da(pred_act_tsr, torch.argmax(a_weights, dim=-1).long(),
                        torch.sigmoid(s_weights).gt(0.5).long(), torch.argmax(c_weights, dim=-1).long())
            pred_act_tsr = pred_act_tsr.ge(1).float()

        c_weight_mat = torch.cat(c_weight_lst, dim=1)
        c_loss = self.c_loss(c_weight_mat.contiguous().view(-1, c_weight_mat.shape[2]),
                             c_target_gold.contiguous().view(-1).long())


        a_weight_mat = torch.cat(a_weight_lst, dim=1)
        a_loss = self.a_loss(a_weight_mat.contiguous().view(-1, a_weight_mat.shape[2]),
                             a_target_gold.contiguous().view(-1).long())

        s_loss /= loss_num
        return c_loss, a_loss, s_loss, pred_act_tsr
