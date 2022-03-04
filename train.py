import torch
import json
import os
import os.path
import math
import glob
import re

import time
import copy
from random import sample
import pytorch_lightning as pl
import random
from pytorch_lightning import Trainer, seed_everything
from utils.dataloader import get_data_loaders, get_current_task_data, make_loader
from utils.opt import parse_train_opt
from utils.utils import make_check_folder

from test import test_model_seq2seq, generate_sample_prev_task, test_model_seq2seq_ADAPTER
from collections import defaultdict
from CL_learner import Seq2SeqToD

from argparse import ArgumentParser


def get_checkpoint(log_dir, index_to_load):
    file = glob.glob(f"{log_dir}/*")
    for f in file:
        f_noprefix = f.replace(f"{log_dir}","")
        num = [int(s) for s in re.findall(r'\d+', f_noprefix)]
        if index_to_load in num:
            version = os.listdir(f+"/lightning_logs")[0]
            check_name = os.listdir(f+"/lightning_logs/"+ version+"/checkpoints/")[0]
            # checkpoint_name = f.replace("[","\[").replace("]","\]").replace("\'","\\'")+"/lightning_logs/"+ version+"/checkpoints/"+check_name
            checkpoint_name = f+"/lightning_logs/"+ version+"/checkpoints/"+check_name
    return checkpoint_name

def load_all_test_loaders(hparams, tokenizer, mode):
    args = copy.deepcopy(hparams)
    if mode == "mutli":
        args.multi, args.continual = True, False
        _, _, test_loader, (_, _, _) = get_data_loaders(args, tokenizer, test=True) # only load all test datasets
    else:
        args.multi, args.continual = False, True
        _, _, test_loader, (_, _, _) = get_data_loaders(args, tokenizer, test=True) # only load all test datasets

    return test_loader

def train(hparams, *args):
    if(hparams.CL == "ADAPTER"):
        hparams.saving_dir = f"/data/wangcong/CL-dialogue/runs/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    else:
        hparams.saving_dir = f"/data/wangcong/CL-dialogue/runs/{hparams.dataset_list}/{hparams.CL}_EM_{hparams.episodic_mem_size}_LAMOL_{hparams.percentage_LAM0L}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    # 如果对数据集进行抽样的话
    if(hparams.sample_dataset_radio != 1.0):
        if(hparams.CL == "ADAPTER"):
            hparams.saving_dir = f"/data/wangcong/CL-dialogue/runs_sample_{hparams.sample_dataset_radio}/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
        else:
            hparams.saving_dir = f"/data/wangcong/CL-dialogue/runs_sample_{hparams.sample_dataset_radio}/{hparams.dataset_list}/{hparams.CL}_EM_{hparams.episodic_mem_size}_LAMOL_{hparams.percentage_LAM0L}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"    
    
    if(hparams.CL == "MULTI"): 
        hparams.multi = True
        hparams.continual = False
    else: 
        hparams.multi = False
        hparams.continual = True

    # train!
    model = Seq2SeqToD(hparams)
    train_loader, valid_loader, test_loader, all_train_loaders, all_valid_loaders, \
                    all_test_loaders, (train_datasets, val_datasets, test_datasets)  = get_data_loaders(hparams, model.tokenizer)

    seed_everything(hparams.seed)

    # do not need
    # ## make the permutation 
    # if(hparams.continual):
    #     keys =  list(train_loader.keys())
    #     random.shuffle(keys)
    #     train_loader = {key: train_loader[key] for key in keys}
    #     print(f"RUNNING WITH SEED {hparams.seed}")
    #     for k,_ in train_loader.items():
    #         print(k)
    #     print()


    task_seen_so_far = []
    TASKS = hparams.dataset_list.split(",")

    if(hparams.CL != "MULTI"): model.set_number_of_tasks(len(list(train_loader.keys())))
    if(hparams.CL == "GEM"): model.set_up_gem()

    if hparams.multi:
        start = time.time()
        trainer = Trainer(
                default_root_dir=hparams.saving_dir,
                accumulate_grad_batches=hparams.gradient_accumulation_steps,
                gradient_clip_val=hparams.max_norm,
                max_epochs=hparams.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00, patience=5,verbose=False, mode='min')],
                gpus=[0],
                )
        trainer.fit(model, all_train_loaders, all_valid_loaders)
        end = time.time()
        print ("Time elapsed:", end - start)
        # save the model and tokenizer
        model.model.save_pretrained(f'{hparams.saving_dir}')
        model.tokenizer.save_pretrained(f'{hparams.saving_dir}')

        make_check_folder(f'{hparams.saving_dir}/FINAL')
        # test for all datasets
        for cur_test_task_id, (cur_test_task) in enumerate(TASKS):
            result_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_{cur_test_task}_train_multi_result.txt'
            gt_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_{cur_test_task}_train_multi_gt.txt'
            test_model_seq2seq(hparams, model.model, model.tokenizer, test_loader[cur_test_task], result_path, gt_path)
        
        # TODO: to test all file
        result_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_multiAll_train_multiAll_result.txt'
        gt_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_multiAll_train_multiAll_gt.txt'
        test_model_seq2seq(hparams, model.model, model.tokenizer, all_test_loaders, result_path, gt_path)

    elif hparams.continual:
        
        for cur_train_task_id, (cur_train_task, task_loader) in enumerate(train_loader.items()):
            model.task_list_seen.append(cur_train_task)
 
            task_path = f'{hparams.saving_dir}/{cur_train_task_id}_{cur_train_task}'

            if(hparams.CL == "REPLAY"):
                print(f"Memory Size {len(model.reply_memory)}")
                # task_loader = make_loader(hparams,train_datasets[task_id]+model.reply_memory,model.tokenizer)
                task_loader = make_loader(hparams, tokenizer=model.tokenizer, origin_dataset=train_datasets[cur_train_task], \
                                                    extra_dataset=model.reply_memory, cur_task=cur_train_task)

            # TODO LAMOL method
            if(hparams.CL == "LAMOL"):
                if(current_task_to_load == None or task_num >= current_task_to_load):
                    number_of_sample = hparams.percentage_LAM0L 
                    aug_current_task = get_current_task_data(hparams,train_datasets[task_id],task_id,number_of_sample)
                    print(f"Current {task_id} AUG: {len(aug_current_task)}")
                    aug_data_prev_task = []
                    for task_id_so_far in task_seen_so_far:
                        ## sample data by the LM, priming with [task_id] e.g., [hotel]
                        temp = generate_sample_prev_task(hparams,model.model,model.tokenizer,train_datasets,task_id_so_far,number_of_sample,time=f"{task_num}_{task_id}")
                        print(f"Current {task_id_so_far} AUG: {len(temp)}")
                        aug_data_prev_task += temp
                    ## this task_loader include data generated by the same model
                    # task_loader = make_loader(hparams,train_datasets[task_id]+aug_current_task+aug_data_prev_task,model.tokenizer)
                    task_loader = make_loader(hparams, tokenizer=model.tokenizer, origin_dataset=train_datasets[task_id], \
                                    extra_dataset=aug_current_task+aug_data_prev_task, cur_task=task_id)

            ## CORE
            print()
            print(f"cur_train_task: {cur_train_task}")
            print("---"*20)
            start = time.time()
            trainer = Trainer(
                default_root_dir=task_path,
                accumulate_grad_batches=hparams.gradient_accumulation_steps,
                gradient_clip_val=hparams.max_norm,
                max_epochs=hparams.n_epochs,
                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=True, mode='min')],
                gpus=[0],
                #limit_train_batches=100,
            )
            trainer.fit(model, task_loader, valid_loader[cur_train_task])
            end = time.time()
            print ("Time elapsed:", end - start)
            
            #load best model 这里可以加载 epoches之后的最好的model
            # this model are better if the are runned to they epoch number
            if(hparams.CL != "LAMOL" and hparams.CL != "EWC"):
                # checkpoint = torch.load(trainer.checkpoint_callback.best_model_path) use this if the next doesn't work
                checkpoint = torch.load(trainer.checkpoint_callback.best_model_path, map_location=lambda storage, loc: storage)
                print("load from:", trainer.checkpoint_callback.best_model_path)
                checkpoint['state_dict'] = { k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() }
                model.model.load_state_dict(checkpoint['state_dict'])

            # # testing the model by generating the answers
            # if(hparams.test_every_step):
            #     if(hparams.CL == "ADAPTER"):
            #         test_model_seq2seq_ADAPTER(hparams,model,model.tokenizer,dev_val_loader, test_datasets,time=f"{task_num}_{task_id}")
            #     else:                
            #         test_model_seq2seq(hparams,model.model,model.tokenizer,dev_val_loader, time=f"{task_num}_{task_id}")

            ## END CORE

            model.first_task = False
            ## save some training data into the episodic mem
            if hparams.CL == "AGEM":
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem["all"].append(b)
                    if idx_b==hparams.episodic_mem_size: break
            elif hparams.CL == "REPLAY":
                # in percentage
                model.reply_memory += sample(train_datasets[cur_train_task],min(len(train_datasets[cur_train_task]), hparams.episodic_mem_size))# sample(train_datasets[task_id],min(len(train_datasets[task_id]),int(hparams.episodic_mem_size*len(train_datasets[task_id])))
            else: ## save example per task
                for idx_b, b in enumerate(task_loader):
                    model.episodic_mem[cur_train_task].append(b)
                    if idx_b==hparams.episodic_mem_size: break


            ##### Compute Fisher info Matrix for EWC
            if hparams.CL == "EWC" or hparams.CL =="L2":
                model.model.cpu()
                for n, p in model.model.named_parameters():
                    model.optpar[n] = torch.Tensor(p.cpu().data)
                    model.fisher[n] = torch.zeros(p.size()) #torch.Tensor(p.cpu().data).zero_()

                if hparams.CL == "EWC":
                    for _, batch in enumerate(model.episodic_mem[cur_train_task]):
                        model.model.zero_grad()

                        input_ids, token_type_ids, labels, target_ids, taskname, indexes, attention_masks_2d, \
                                    kg_pad_ids, kg_memory_mask, kg_pad_kn_num = tuple(input_tensor for input_tensor in batch)

                        (loss), *_ = model.model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                labels=labels)

                        loss.backward()
                        for n, p in model.model.named_parameters():
                            if p.grad is not None:
                                model.fisher[n].data += p.grad.data ** 2

                    for name_f,_ in model.fisher.items():
                        model.fisher[name_f] /= len(model.episodic_mem[cur_train_task]) #*hparams.train_batch_size
                    model.model.zero_grad()
            task_seen_so_far.append(cur_train_task)

            # save task model and tokenizer
            model.model.save_pretrained(f'{task_path}')
            model.tokenizer.save_pretrained(f'{task_path}')

            # for model test  
            make_check_folder(f'{hparams.saving_dir}/FINAL')

            
            for cur_test_task_id, (cur_test_task) in enumerate(TASKS[:(cur_train_task_id+1)]):

                print(f"For test: cur_train_task: {cur_train_task}\t cur_test_task: {cur_test_task}")
                result_path = f'{hparams.saving_dir}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_result.txt'
                gt_path = f'{hparams.saving_dir}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_gt.txt'
                
                task_id = cur_train_task_id if (hparams.CL == "ADAPTER") else -1
                test_model_seq2seq(hparams, model.model, model.tokenizer, test_loader[cur_test_task],
                                                        result_path, gt_path, task_id=task_id)

        # TODO: test all 
        result_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_continualAll_train_continualAll_result.txt'
        gt_path = f'{hparams.saving_dir}/FINAL' + f'/multiSkill_test_continualAll_train_continualAll_gt.txt'
        test_model_seq2seq(hparams, model.model, model.tokenizer, all_test_loaders, result_path, gt_path)





if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=0 python train.py --task_type NLG --CL MULTI 
    hyperparams = parse_train_opt()
    train(hyperparams)
