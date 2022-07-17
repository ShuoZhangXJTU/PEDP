import argparse
import random
from args import get_parser
from config_multiwoz import MultiWozConfig
from trainer_base import Trainer
from trainer_gcas import TrainerGCAS
import logging
from utils import init_logging_handler
from best_hyperpara import best_hyperpara
from sgd_data.config import SGDConfig


if __name__ == '__main__':
    args = get_parser()

    if args.pedp:
        model_name = 'pedp'
    elif args.gcas:
        model_name = 'gcas'
    elif args.seq:
        model_name = 'seq'
    elif args.cls:
        model_name = 'cls'
    elif args.md:
        model_name = 'md'
    else:
        raise NotImplementedError('Choose a model by using --name (pedp/gcas/seq/cls/md)')

    args = vars(args)
    for key_, val_ in best_hyperpara[model_name].items():
        args[key_] = val_

    if args['seed'] is None:
        args['seed'] = random.randint(0, 10000)

    args['checkpoint_dir'] += '/{}_{}'.format(model_name, args['name'])
    args['log_dir'] += '/{}_{}'.format(model_name, args['name'])
    args['tb_dir'] += '/{}_{}'.format(model_name, args['name'])
    args = argparse.Namespace(**args)

    if args.debug:
        init_logging_handler(args.log_dir, level='debug')
    else:
        init_logging_handler(args.log_dir, level='info')

    logging.info('args settings: {}'.format(str(args)))

    if args.sgd:
        cfg = SGDConfig()
    else:
        cfg = MultiWozConfig()

    cfg.temperature = args.temperature
    cfg.data_ratio = args.data_ratio
    cfg.dropout = args.dropout
    cfg.h_dim = args.h_dim

    if args.gcas:
        trainer = TrainerGCAS(args, cfg)
    else:
        trainer = Trainer(args, cfg)

    logging.info('start imitation')
    for epoch in range(args.epoch):
        trainer.imitate(epoch)
