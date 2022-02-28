import torch
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from functools import partial
from utils.preprocess import get_datasets
from collections import defaultdict
import pprint
import random
from tabulate import tabulate

# from dataset_ms import MSDataset
from utils.dataset_ms import MSDataset

pp = pprint.PrettyPrinter(indent=4)


class DatasetTrain(Dataset):
    """Custom data.Dataset compatible with DataLoader."""
    def __init__(self, data, domains=None):
        self.data = data
        self.dataset_len = len(self.data)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        item = self.data[index]
        return item

    def __len__(self):
        return self.dataset_len



def get_from_dial(args, data, task_name, tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        dialogues.append({"history":latest_API_OUT,
                        "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                        "history_reply": latest_API_OUT + f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                        "spk":t["spk"],
                        "dataset":t["dataset"],
                        "dial_id":t["id"],
                        "turn_id":t["turn_id"],
                        "task_id":task_id})

    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues

def get_e2e_from_dial(args,data,task_id,tokenizer):
    dialogues = []
    utt_len = []
    hist_len = []
    for dial in data:
        plain_history = []
        latest_API_OUT = "API-OUT: "
        for idx_t, t in enumerate(dial['dialogue']):
            ## DUPLICATE DIALOGUE
            if f'{t["id"]}' == "dlg-ff2b8de2-467d-4917-be13-1529765752e9" and f'{t["id"]}' == "dlg-fdd242eb-56be-48c0-a56e-5478472500d0":
                continue
            if(t['spk']=="USER"):
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
            elif(t['spk']=="API-OUT"):
                latest_API_OUT = f"{t['spk']}: {t['utt'].strip()}"
            elif((t['spk'] == "SYSTEM") and idx_t!=0 and t["utt"]!= ""):
                dialogues.append({"history":" ".join(plain_history[-args.max_history:] + [latest_API_OUT]),
                                  "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:] + [latest_API_OUT])+ f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":t["turn_id"],
                                  "task_id":task_id})
                plain_history.append(f"{t['spk']}: {t['utt'].strip()}")
                latest_API_OUT = "API-OUT: "
            elif((t['spk'] == "API") and idx_t!=0 and t["utt"]!= ""):
                dialogues.append({"history":" ".join(plain_history[-args.max_history:]),
                                  "reply":f'{t["utt"].strip()} {tokenizer.eos_token}',
                                  "history_reply": " ".join(plain_history[-args.max_history:])+ f'[SOS]{t["utt"].strip()} {tokenizer.eos_token}',
                                  "spk":t["spk"],
                                  "dataset":t["dataset"],
                                  "dial_id":t["id"],
                                  "turn_id":str(t["turn_id"])+"API",
                                  "task_id":task_id})
    if args.verbose:
        for d in random.sample(dialogues,len(dialogues)):
            pp.pprint(d)
            break
        print()
        input()
    return dialogues


def get_current_task_data(args,dataset_dic,task_id,number_of_sample):
    temp_aug = random.sample(dataset_dic,min(number_of_sample,len(dataset_dic)))
    aug_data = []
    cnt_API = 0
    for d in temp_aug:
        ## add a first token for the generation
        if(args.task_type=="E2E"):
            if(d["spk"]=="API"):
                cnt_API += 1
                d["history_reply"] = f"[{str(eval(task_id)[0])}-API]"+d["history_reply"]
            else:
                d["history_reply"] = f"[{str(eval(task_id)[0])}]"+d["history_reply"]
        else:
            d["history_reply"] = f"[{str(eval(task_id)[0])}]"+d["history_reply"]
        aug_data.append(d)
    return aug_data


def collate_fn(data,tokenizer):
    batch_data = {}
    

    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = input_batch["attention_mask"]
    output_batch = tokenizer(batch_data["reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']
    return batch_data

def collate_fn_GPT2(data, tokenizer):
    batch_data = {}


    for key in data[0]:
        batch_data[key] = [d[key] for d in data]

    input_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False,return_attention_mask=False)
    batch_data["encoder_input"] = input_batch["input_ids"]
    batch_data["attention_mask"] = None
    output_batch = tokenizer(batch_data["history_reply"], padding=True, return_tensors="pt", truncation=False,add_special_tokens=False, return_attention_mask=False)
    # replace the padding id to -100 for cross-entropy
    output_batch['input_ids'].masked_fill_(output_batch['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["decoder_output"] = output_batch['input_ids']

    #### DATA FOR COMPUTING PERPLEXITY OF DIALOGUE HISTORY ==> FOR ADAPTER
    batched_history = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False,return_attention_mask=False)
    batch_data["input_id_PPL"] = batched_history['input_ids']
    batched_history_out = tokenizer(batch_data["history"], padding=True, return_tensors="pt", truncation=False, add_special_tokens=False, return_attention_mask=False)
    batched_history_out['input_ids'].masked_fill_(batched_history_out['input_ids']==tokenizer.pad_token_id, -100)
    batch_data["output_id_PPL"] = batched_history_out['input_ids'] ### basically just remove pad from ppl calculation
    return batch_data


def make_loader(args,list_sample,tokenizer):
    collate_fn_ = collate_fn_GPT2 if("gpt2" in args.model_checkpoint) else collate_fn
    return DataLoader(DatasetTrain(list_sample), batch_size=args.train_batch_size, shuffle=True,collate_fn=partial(collate_fn_, tokenizer=tokenizer))

def get_data_loaders(args, tokenizer, test=False):
    """ Prepare the dataset for training and evaluation """
    aggregate = get_datasets(dataset_list=args.dataset_list.split(','), setting=args.setting, verbose=args.verbose, develop=args.debug)

    collate_fn_ = collate_fn_GPT2 if("gpt2" in args.model_checkpoint) else collate_fn
    
    if(test):
        datasets = {"test":{}}
    else:
        datasets = {"train":{}, "dev": {}, "test":{}}

    for split in datasets.keys():
        for task_name, taskDict in aggregate["AllDatasets"].items():
            datasets[split][task_name] = taskDict[split]
            # datasets[split][task_name] = get_from_dial(args, taskDict[split], task_name, tokenizer)


    ### LOGGING SOME INFORMATION ABOUT THE TASKS
    print(f"All Datasets: {aggregate['AllDatasets'].keys()}")
    print(f"Num of Tasks {len(aggregate['AllDatasets'].keys())}")

    task = defaultdict(lambda: defaultdict(str))
    for split in ["train","dev","test"]:
        for task_name, dataset_task in datasets[split].items():
            task[task_name][split] = len(dataset_task)

    table = []
    for task_name, split_len in task.items():
        table.append({"task":task_name, "train": split_len["train"], "dev": split_len["dev"], "test": split_len["test"]})
    print(tabulate(table, headers="keys"))

    train_loaders = {}
    valid_loaders = {}

    train_datasets = {}
    val_datasets = {}
    if(args.continual):
        if(not test):
            for task_id, dataset_task in datasets["train"].items():
                trainMSDatasets = MSDataset(dataset_task, tokenizer, cur_task=task_id+"_train")
                train_loaders[task_id] = DataLoader(
                    trainMSDatasets, 
                    batch_size=args.train_batch_size, 
                    collate_fn=validMSDatasets.collate,
                    num_workers=args.num_workers,
                    shuffle=True)
                train_datasets[task_id] = dataset_task
            for task_id, dataset_task in datasets["dev"].items():
                validMSDatasets = MSDataset(dataset_task, tokenizer, cur_task=task_id+"_dev")
                valid_loaders[task_id] = DataLoader(
                    validMSDatasets, 
                    batch_size=args.valid_batch_size,
                    collate_fn=validMSDatasets.collate,
                    num_workers=args.num_workers,
                    shuffle=False)
                val_datasets[task_id] = dataset_task
    
    elif(args.multi):
        if(not test):
            dataset_train = []
            for task_id, dataset_task in datasets["train"].items():
                dataset_train += dataset_task 
            trainMSDatasets = MSDataset(dataset_train, tokenizer, cur_task="_".join(aggregate['AllDatasets'].keys())+"_train")
            train_loaders = DataLoader(
                trainMSDatasets, 
                batch_size=args.train_batch_size, 
                collate_fn=trainMSDatasets.collate,
                num_workers=args.num_workers,
                shuffle=True)

            dataset_dev = []
            for task_id, dataset_task in datasets["dev"].items():
                dataset_dev += dataset_task
            validMSDatasets = MSDataset(dataset_dev, tokenizer, cur_task="_".join(aggregate['AllDatasets'].keys())+"_dev")
            valid_loaders = DataLoader(
                validMSDatasets, 
                batch_size=args.valid_batch_size,
                num_workers=args.num_workers,
                collate_fn=validMSDatasets.collate,
                shuffle=False,)

    # TODO: whether need to modify
    temp_list = []
    for task_id, dataset_task in datasets["test"].items():
        temp_list.append(dataset_task)
    test_datasets = sum(temp_list,[])
    testMSDatasets = MSDataset(test_datasets, tokenizer, cur_task="_".join(aggregate['AllDatasets'].keys()) + "_test", lm_labels=False, with_eos=False)
    test_loaders = DataLoader(
        DatasetTrain(sum(temp_list,[])), 
        batch_size=args.test_batch_size, # test_batch_size = 1
        collate_fn=testMSDatasets.collate, # unless to get padding result
        num_workers=args.num_workers,
        shuffle=False)



    return train_loaders, valid_loaders, test_loaders, (train_datasets, val_datasets, test_datasets)
