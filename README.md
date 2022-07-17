# PEDP
“Think Before You Speak”: Improving Multi-action Dialog Policy by Planning Single-Action Dialogs  (IJCAI-22 long oral presentation)

<h1>Planning-Enhanced Dialog Policy</h1>
This is the codebase for the proposed multi-action 
dialog policy model PEDP and all the SL-based baseline models (gCAS, DiaMultiClass, DiaMultiDense, and DiaSeq).

For other models, we refer readers to the official implementations. ([GDPL](https://github.com/truthless11/GDPL) and [DiaAdv](https://github.com/cszmli/Rethink-RL-Sup))

---

<h3>Requirements</h3>
Please refer to <code>environment.yml</code> 
and prepare the environment with Anaconda.

---

<h3>Data</h3>
We report results on: 
1. [MultiWOZ](https://arxiv.org/abs/1810.00278).
Please download the data from [here](https://drive.google.com/file/d/1r8-1h7jyolR5lsETP0TxJty4UIKESN-d/view?usp=sharing) and unzip under <code>./data</code> directory.
2. [SGD](https://arxiv.org/abs/1909.05855). Please download the data from [here](https://drive.google.com/file/d/1I-5w4CTsYnOrFm1TMMrc62OWtcBq3vAm/view?usp=sharing) and unzip under <code>./sgd_data</code> directory.

---

<h3>Run</h3>
To reproduce the results on MultiWOZ, execute:
```
python -u main.py --pedp --residual
```

To reproduce the results on SGD, execute:
```
python -u main.py --pedp --residual --sgd
```

To train other models, execute:
```
python -u main.py --[model_name]
```

More hyper-parameters are assigned in <code>args.py</code>
and can be modified using <code>--para=value</code>.

Dataset schema is defined in <code>config_multiwoz.py</code> and <code>./sgd_data/config.py</code> for MultiWOZ and SGD, respectively.



---

<h3>Results</h3>
We strongly recommend using tensorboard to check the results.
Execute:
```
tensorboard --bind_all --logdir=./log/tb/[file name]
```
and open the corresponding website.
