# Continual Learning for Task-Oriented Dialogue Systems

This repository includes the dataset and baselines of the paper:

**Continual Learning for Task-Oriented Dialogue Systems** (Accepted in EMNLP 2021) [[PDF]](https://arxiv.org/abs/2012.15504). 

**Authors**: [Andrea Madotto](https://andreamad8.github.io), [Zhaojiang Lin](https://zlinao.github.io), Zhenpeng Zhou, [Seungwhan Moon](https://shanemoon.com/), [Paul Crook](http://pacrook.net/), [Bing Liu](http://bingliu.me/), [Zhou Yu](https://www.cs.columbia.edu/~zhouyu/), Eunjoon Cho, [Zhiguang Wang](https://research.fb.com/people/wang-zhiguang/), [Pascale Fung](https://pascale.home.ece.ust.hk/)


## Abstract
Continual learning in task-oriented dialogue systems allows the system to add new domains and functionalities over time after deployment, without incurring the high cost of retraining the whole system each time. In this paper, we propose a first-ever continual learning benchmark for task-oriented dialogue systems with 37 domains to be learned continuously in both modularized and end-to-end learning settings.  In addition, we implement and compare multiple existing continual learning baselines, and we propose a simple yet effective architectural method based on residual adapters. We also suggest that the upper bound performance of continual learning should be equivalent to multitask learning when data from all domain is available at once. Our experiments demonstrate that the proposed architectural method and a simple replay-based strategy perform better, by a large margin, compared to other continuous learning techniques, and only slightly worse than the multitask learning upper bound while being 20X faster in learning new domains. We also report several trade-offs in terms of parameter usage, memory size and training time, which are important in the design of a task-oriented dialogue system. The proposed benchmark is released to promote more research in this direction.  

## Installation
The Continual Learning benchmark is created by jointly pre-processing four task-oriented dataset such as [Task-Master (TM19)](https://github.com/google-research-datasets/Taskmaster.git), [Task-Master 2020 (TM20)](https://github.com/google-research-datasets/Taskmaster.git), [Schema Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.git) and [MultiWoZ](https://github.com/budzianowski/multiwoz.git). To download the dataset, and setup basic package use: 
```
pip install -r requirements.txt
cd data
bash download.sh
```
If you are interested in the pre-processing, please check ```utils/preprocess.py``` and ```utils/dataloader.py```.

## Basic Running
In this codebase, we implemented several baselines such as MULTI, VANILLA, L2, EWC, AGEM, LAMOL, REPLAY, ADAPTER, and four ToDs settings such as INTENT, DST, NLG, E2E. An example for running the NLG task with a VANILLA method is:  
```
CUDA_VISIBLE_DEVICES=0 python train.py --CL VANILLA --task_type NLG
```
Different CL methods uses different hyperparamters. For example, in REPLAY you can tune the episodic memory size as following: 
```
CUDA_VISIBLE_DEVICES=0 python train.py --CL REPLAY --episodic_mem_size 10
```
this will randomly sample 10 example per task, and replay it while learning new once. A full example to run the baseline is for example: 

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py --CL REPLAY --episodic_mem_size 100 > ./outputs/alldatasets/REPLAY.log 2>&1 &

```
CUDA_VISIBLE_DEVICES=0 nohup python train.py --CL REPLAY --episodic_mem_size 100 > ./outputs/alldatasets/REPLAY.log 2>&1 &


CUDA_VISIBLE_DEVICES=1 nohup python train.py --CL MULTI  > ./outputs/alldatasets/MULTI.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python train.py --CL VANILLA   > ./outputs/alldatasets/VANILLA.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train.py --CL L2 --reg 0.01   > ./outputs/alldatasets/L2.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python train.py --CL EWC --reg 0.01  > ./outputs/alldatasets/EWC.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python train.py --CL AGEM --episodic_mem_size 100 --reg 1.0  > ./outputs/alldatasets/AGEM.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python train.py --CL LAMOL --percentage_LAM0L 200  > ./outputs/alldatasets/LAMOL.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python train.py --CL ADAPTER --bottleneck_size 75 --lr 6.25e-3 > ./outputs/alldatasets/ADAPTER.log 2>&1 &
```


## Hyperparameters

### Adapter
```
python train.py --CL ADAPTER --bottleneck_size 300 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --CL ADAPTER --bottleneck_size 50 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
python train.py --CL ADAPTER --bottleneck_size 100 --lr 6.25e-3 --n_epochs 10 --train_batch_size 10 --gradient_accumulation_steps 8
```

### Replay
```
python train.py --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
python train.py --CL REPLAY --episodic_mem_size 50 --lr 6.25e-5 --n_epochs 10 --train_batch_size 8 --gradient_accumulation_steps 8
```


## Evaluation 

```
python scorer.py --model_checkpoint runs/BEST/
python scorer.py --model_checkpoint /data/wangcong/CL-dialogue/runs/Convai2,Ed,Wow,Daily,Cornell/

```

### Modularized

| Name    |    INTENT |       JGA |     BLEU |       EER |
|---------|-----------|-----------|----------|-----------|
| VANILLA | 0.0303205 | 0.102345  | 10.3032  | 0.181644  |
| L2      | 0.0346528 | 0.0923626 | 11.0159  | 0.189819  |
| EWC     | 0.0283001 | 0.0998913 |  9.65351 | 0.203158  |
| AGEM    | 0.102224  | 0.0965043 |  4.61297 | 0.360167  |
| LAML    | 0.0262127 | 0.0923302 |  3.49649 | 0.35664   |
| REPLAY  | 0.800088  | 0.394993  | 21.4832  | 0.0559855 |
| ADAPTER | 0.841951  | 0.37381   | 21.7719  | 0.163975  |
| MULTI   | 0.875002  | 0.500357  | 26.1462  | 0.0341823 |


### NLG

| Name    |     BLEU |       EER |
|---------|----------|-----------|
| MULTI   | 26.1462  | 0.0341823 |
| VANILLA | 10.3032  | 0.181644  |
| AGEM    |  4.61297 | 0.360167  |
| L2      | 10.5389  | 0.189819  |
| REPLAY  | 20.5041  | 0.0738263 |
| ADAPTER | 21.7719  | 0.163975  |
| EWC     |  9.65351 | 0.203158  |
| LAML    |  3.49649 | 0.35664   |



# Acknowledgement
I would like to thanks [Saujas Vaduguru](saujas.vaduguru@mila.quebec), [Qi Zhu](zhuq96@gmail.com), and [Maziar Sargordi](maziar.sargordi@mila.quebec) for helping with debugging the code. 
