import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    # -- trail information
    parser.add_argument('--name', type=str, default='base')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--cpus', type=int, default=6)
    parser.add_argument('--trail', type=int, default=1)

    parser.add_argument('--sgd', default=False, action='store_true')

    # -- log settings
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--log_interval', type=int, default=10)

    # -- global train hyper-parameters
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batchsz', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--data_ratio', type=int, default=100, help='data size ratio')

    # -- choose models
    parser.add_argument('--pedp', default=False, action='store_true')
    parser.add_argument('--seq', default=False, action='store_true')
    parser.add_argument('--cls', default=False, action='store_true')
    parser.add_argument('--md', default=False, action='store_true')
    parser.add_argument('--gcas', default=False, action='store_true')

    # -- PEDP
    parser.add_argument('--residual', default=False, action='store_true')
    parser.add_argument('--paths', type=int, default=3)
    parser.add_argument('--aggr_type', type=str, default='avg')
    parser.add_argument('--pred_gamma', type=float, default=1e+1)
    parser.add_argument('--plan_gamma', type=float, default=1e-1)
    parser.add_argument('--state_gamma', type=float, default=1e-1)
    parser.add_argument('--term_gamma', type=float, default=1e-1)

    # -- Seq2seq
    parser.add_argument('--beam', default=False, action='store_true')

    # -- Action Sampling
    parser.add_argument('--gumbel', default=False, action='store_true')
    parser.add_argument('--tau_plan_a', type=float, default=1e-3)
    parser.add_argument('--tau_plan_t', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1e-3)

    # -- Dirs
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--log_dir', type=str, default='./log/text')
    parser.add_argument('--tb_dir', type=str, default='./log/tb')
    parser.add_argument('--data_dir', type=str, default='./data/multiwoz')
    parser.add_argument('--nlg_dir', type=str, default='./nlg/template/multiwoz')

    # -- Teacher Forcing Decay
    parser.add_argument('--beta_upper', type=float, default=1)
    parser.add_argument('--beta_decay', type=float, default=36)
    parser.add_argument('--beta_lower', type=float, default=0.25)

    return parser.parse_args()
