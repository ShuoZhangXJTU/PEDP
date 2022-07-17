import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler, ASHAScheduler
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from para_search.search_space import get_space, get_resources
from ray.tune import Analysis
from para_search.search_train_func import train_for_turn
from args import get_parser


class ParaSearcher:
    def __init__(self):
        self.server = get_resources()
        ray.shutdown()
        ray.init(num_gpus=self.server.gpu,
                 num_cpus=self.server.cpu,
                 dashboard_host='0.0.0.0')
                 # dashboard_port=6666)
        self.args = get_parser()
        self.args = vars(self.args)
        search_space, current_best_params, _ = get_space('pedp')
        for key_, val_ in search_space.items():
            self.args[key_] = val_

        self.search_alg = HyperOptSearch(
            random_state_seed=6666,
            # points_to_evaluate=current_best_params
         )
        self.search_alg = ConcurrencyLimiter(self.search_alg, max_concurrent=128)
        self.scheduler = AsyncHyperBandScheduler(grace_period=36)
        # self.scheduler = ASHAScheduler(grace_period=16)

    def search(self):
        # analysis = tune.run(train_for_turn,
        #                     metric='success',
        #                     mode='max',
        #                     config=self.args,
        #                     resources_per_trial={"gpu": 0,
        #                                          "cpu": 1},
        #                     )
        # num_samples=self.server.num_samples,
        # search_alg=self.search_alg)
        #
        analysis = tune.run(train_for_turn,
                            scheduler=self.scheduler,
                            search_alg=self.search_alg,
                            metric="success",
                            mode='max',
                            num_samples=1024,
                            config=self.args,
                            resources_per_trial={"gpu": 0,
                                                 "cpu": 1})

        print("Best config: ", analysis.get_best_config(metric='success', mode='max'))

def analysis(model_name):
    space_dict, _, keys_search = get_space(model_name)
    dir_ = 'ray_results_0821_9'

    dir_ = 'ray_results_0903_md_retest_1'
    rst = Analysis('/data2t/szhangspace/{}/train_for_turn'.format(dir_))
    # rst = Analysis('/home/szhang/{}/train_for_turn'.format(dir_))
    print('=' * 60)
    print('Analyzing Tune Result: {}'.format(dir_))
    print('-' * 60)

    rst_pd = rst.dataframe(metric='success', mode='max')
    rst_pd = rst_pd.sort_values(by='success', ascending=False)
    rst_pd.to_csv('tune_{}.csv'.format(model_name))

    print(rst_pd)
    # select_cols = ['success', 'trial_id', 'training_iteration', 'inform_f1', 'inform_R', 'match', 'turn']
    # select_cols.extend(keys_search)
    # rst_pd = rst_pd[select_cols]
    # rst_pd.to_csv('tune_{}.csv'.format(model_name))
    # print(rst_pd)


