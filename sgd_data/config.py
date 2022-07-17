from .ref import *


class Config:
    def __init__(self):
        self.domain = []
        self.intent = []
        self.slot = []
        self.da = []
        self.da_usr = []
        self.data_file = ''
        self.db_domains = []
        self.belief_domains = []

    def init_inform_request(self):
        self.inform_da = []
        self.request_da = []
        self.requestable = {}
        for domain in self.domain:
            self.requestable[domain] = []

        for da in self.da_usr:
            if len(da.split('-')) == 2:
                continue
            d, i, s = da.split('-')
            if 'INFORM' in i:
                self.inform_da.append('-'.join([d, s]))
            elif 'REQUEST' in i:
                self.request_da.append('-'.join([d, s]))
                self.requestable[d].append(s)

    def init_dict(self):
        self.domain2idx = dict((a, i) for i, a in enumerate(self.domain))
        self.idx2domain = dict((v, k) for k, v in self.domain2idx.items())

        self.intent2idx = dict((a, i) for i, a in enumerate(self.intent))
        self.idx2intent = dict((v, k) for k, v in self.intent2idx.items())

        self.slot2idx = dict((a, i) for i, a in enumerate(self.slot))
        self.idx2slot = dict((v, k) for k, v in self.slot2idx.items())

        self.inform2idx = dict((a, i) for i, a in enumerate(self.inform_da))
        self.idx2inform = dict((v, k) for k, v in self.inform2idx.items())

        self.request2idx = dict((a, i) for i, a in enumerate(self.request_da))
        self.idx2request = dict((v, k) for k, v in self.request2idx.items())

        self.da2idx = dict((a, i) for i, a in enumerate(self.da))
        self.idx2da = dict((v, k) for k, v in self.da2idx.items())

        self.dau2idx = dict((a, i) for i, a in enumerate(self.da_usr))
        self.idx2dau = dict((v, k) for k, v in self.dau2idx.items())

    def init_dim(self):
        # self.s_dim = len(self.da_usr) + len(self.inform_da) + len(self.request_da)
        self.s_dim = len(self.da) * 2 + len(self.da_usr) * 2 + len(self.inform_da) + len(self.request_da) + 1
        self.a_dim = len(self.da)
        self.domain_dim = len(self.domain)
        self.a_dim_usr = len(self.da_usr)


class SGDConfig(Config):
    def __init__(self):
        self.domain = DOMAINS

        self.domain_index = dict(zip(DOMAINS, range(len(DOMAINS))))

        self.intent = INTENTS

        self.slot = set()
        for cur_set in SLOTS.values():
            self.slot = self.slot | cur_set
        self.slot = list(self.slot)

        self.da = set()
        for cur_set in DA_FILTER.values():
            self.da = self.da | cur_set
        self.da = list(self.da)

        self.action2domain = []
        for a in self.da:
            domain = a.strip().split('-')[0]
            self.action2domain.append(self.domain_index[domain])

        self.da_usr = set()
        for cur_set in DAU_FILTER.values():
            self.da_usr = self.da_usr | cur_set
        self.da_usr = list(self.da_usr)

        self.init_inform_request()  # call this first!
        self.init_dict()
        self.init_dim()

        self.mt_factor = 0.2
        self.h_dim = 200
        self.hv_dim = 50  # for value function
        self.hu_dim = 200  # for user module
        self.dataset_dir = 'sgd_data/sgd/processed/'
        self.max_ulen = 20
        self.alpha = 0.01
        self.hi_dim = 50  # for airl module
        self.max_len = 10
        self.embed_size = 64
