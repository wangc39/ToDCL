# -*- coding: utf-8 -*-
import os
import re
import numpy as np
from itertools import chain
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = {"Convai2": ["[bos]", "[eos]", "[speaker1]", "[speaker2]", "[bg]" ],
                  "Daily": ["[bos]", "[eos]", "[speaker1]", "[speaker2]", "[bg]"], 
                  "Ed": ["[bos]", "[eos]", "[speaker1]", "[speaker2]", "[bg]" ],
                  "Wow": ["[bos]", "[eos]", "[speaker1]", "[speaker2]", "[bg]"],
                  }

ATTR_TO_SPECIAL_TOKEN = {'pad_token': '[PAD]', 'unk_token': '[unk]',
                    'additional_special_tokens': 
                    ("[bos]", "[eos]", "[speaker1]", "[speaker2]", "[bg]", 
                    "Convai2", "[Ed]", "[Cornell]", "[Ubuntu]", "[Daily]", "[Wow]")} # add for task name


class MSDataset(Dataset):

    def __init__(self, data, tokenizer, extra_data=None, cur_task=None, max_history=15, max_seq_len=512, max_response_len=128, batch_first=True, \
        lm_labels=True, with_eos=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.max_seq_len = max_seq_len
        self.max_response_len = max_response_len

        self.pad = tokenizer.pad_token_id
        # print(tokenizer.pad_token_id)
        # print(tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['pad_token']))

        # print("tokenizer.eos_token_id", tokenizer.eos_token_id)
        # print("tokenizer.bos_token_id", tokenizer.bos_token_id)
        # print("SPECIAL_TOKENS: ", tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["Convai2"]))
        # print("tokenizer.eos_token_id", tokenizer.eos_token_id)

        # print("====="*20)
        assert(self.pad == self.tokenizer.convert_tokens_to_ids(ATTR_TO_SPECIAL_TOKEN['pad_token']))

        self.batch_first = batch_first # 表示第一维度 是否是batch_size对应的维度一般是True
        # 当模型进行test的时候 lm_labels和eos为False
        self.lm_labels = lm_labels
        self.with_eos = with_eos
        self.cur_task = cur_task if cur_task else None # When multi task, cur task is uncertain

        # load dataset from token
        dataset_cache_path = "cache/{}_{}".format(self.cur_task, type(self.tokenizer).__name__)
        if extra_data:
            extra_data_token = self.get_data_token(dataset_cache_path=None)
            data = self.get_data_token(dataset_cache_path=dataset_cache_path)
            data = data + extra_data_token
        else:
            data = self.get_data_token(dataset_cache_path=dataset_cache_path)
        self.data = data
    
    def get_data_token(self, dataset_cache_path):
        
        data = defaultdict(list)
        if dataset_cache_path and os.path.exists(dataset_cache_path):
            print("Loading dataset from ({})".format(dataset_cache_path))
            data = torch.load(dataset_cache_path)
        else:
            print("Set up dataset cache, cache file path ({})".format(dataset_cache_path))    
            # recursion function
            def tokenize(obj):
                if isinstance(obj, str):
                    return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(obj))
                if isinstance(obj, dict):
                    return dict((n, tokenize(o)) for n, o in obj.items())
                if isinstance(obj, int):
                    return obj
                return list(tokenize(o) for o in obj)
        
            data = tokenize(self.data)
            if dataset_cache_path:
                torch.save(data, dataset_cache_path)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
         
        # print(f"self.cur_task: {self.cur_task}")
        instance = None
        # print("task name", self.tokenizer.convert_ids_to_tokens(self.data[index]["dataset"]))
        # to get cur task name
        if self.data[index]["dataset"]:
            self.cur_task = re.sub(r"[^a-zA-Z]", "", self.tokenizer.convert_ids_to_tokens(self.data[index]["dataset"])[0])

        # print(type(self.cur_task), self.cur_task)
        # self.cur_task = self.data[index]["dataset"] if self.data[index]["dataset"] else self.cur_task 
        
        if self.cur_task == "Convai2":
            # TODO: add speaker1 and speaker2 to one list
            # to get speaker1 and speaker2
            speaker1 = self.data[index]["speaker1"].copy() # four utterances shape: [[], [], [], []]
            speaker2 = self.data[index]["speaker2"].copy() # four utterances

            # speaker1 = self.data[index]["kg"][:4].copy() # four utterances shape: [[], [], [], []]
            # speaker2 = self.data[index]["kg"][4:].copy() # four utterances

            persona = speaker1.copy()
            persona.extend(speaker2)
            length = len(persona)
            for i, utterance in enumerate(self.data[index]["utterances"]):
                persona = persona[(i % length): ] + persona[: (i % length)] # permute persona
                history = utterance["history"][-(2*self.max_history + 1):]
                candidate = utterance["candidates"][-1] if self.lm_labels else [] # only last one candidate which is gold response
                instance = self.proprecess_convai2(index, persona, history, candidate, self.tokenizer, task=self.cur_task)
                if not self.lm_labels: instance["target"] = utterance["candidates"][-1] # add for data test

        elif self.cur_task == "Daily":
            topic = self.data[index]["topic"]
            for i, utterance in enumerate(self.data[index]["utterances"]):
                history = utterance["history"][-(2*self.max_history + 1):]
                candidate = utterance["candidates"][-1] if self.lm_labels else [] # only have one sentence
                instance = self.proprecess_daily(index, topic, history, candidate, self.tokenizer, task=self.cur_task)
                if not self.lm_labels: instance["target"] = utterance["candidates"][-1] # add for data test


        elif self.cur_task == "Wow":
            topic = self.data[index]["topic"]
            persona = self.data[index]["persona"] # one utterance
            for i, utterance in enumerate(self.data[index]["utterances"]):
                history = utterance["history"][-(2*self.max_history + 1):]
                candidate = utterance["candidates"][-1] if self.lm_labels else [] # only have one sentence
                instance = self.proprecess_wow(index, topic, persona, history, candidate, self.tokenizer, task=self.cur_task)
                if not self.lm_labels: instance["target"] = utterance["candidates"][-1] # add for data test


        elif self.cur_task == "Ed":
            context = self.data[index]["context"] # empathic word
            prompt = self.data[index]["prompt"] 
            for i, utterance in enumerate(self.data[index]["utterances"]):
                history = utterance["history"][-(2*self.max_history + 1):]
                candidate = utterance["candidates"][-1] if self.lm_labels else [] # only have one sentence
                instance = self.proprecess_ed(index, context, prompt, history, candidate, self.tokenizer, task=self.cur_task)
                if not self.lm_labels: instance["target"] = utterance["candidates"][-1] # add for data test


        elif self.cur_task == "Ubuntu" or self.cur_task == "cornell": # ubuntu and cornell task is same
            for i, utterance in enumerate(self.data[index]["utterances"]):
                history = utterance["history"][-(2*self.max_history + 1):]
                candidate = utterance["candidates"][-1] if self.lm_labels else [] # only have one sentence
                instance = self.proprecess_ubuntu(index, history, candidate, self.tokenizer, task=self.cur_task)
                if not self.lm_labels: instance["target"] = utterance["candidates"][-1] # add for data test

        else:
            print(f"current task: {self.cur_task}")
            raise Exception("Can not find current task")


        return instance

    def _truncate_list(self, ids_list, cut_len=512, truncate_first_turn=False):
        if sum([len(x) for x in ids_list]) <= cut_len:
            return ids_list
        
        new_ids_list = []
        ids_list.reverse()
        len_cnt = 0

        for  i, ids in enumerate(ids_list):
            if len_cnt + len(ids) > cut_len:
                if len_cnt == 0 and (len(ids_list) > 1 or not truncate_first_turn): # last utterance of context is too long
                    new_ids_list.append(ids[-cut_len:])
                    len_cnt = cut_len
                elif truncate_first_turn and i == len(ids_list) - 1 and len_cnt + 1 < cut_len: # first utterance of context is too long and trunc
                    new_ids_list.append(ids[:cut_len - len_cnt - 1] + [ids[-1]])
                    len_cnt = cut_len
                else:
                    pass # other utterance of context is too long and trunc
                break
            else:
                len_cnt += len(ids)
                new_ids_list.append(ids)

        new_ids_list.reverse()
        return new_ids_list

    def proprecess_convai2(self, index, persona, history, response, tokenizer, task):
        """
        Args:
            persona ([[], [], [], []]): a list represent the persona information
            hisory (two dim list): a list represent the conservation
            response(one dim list): a list
            task (a string) : task name
        """
        bos, eos, speaker1, speaker2, bg_token = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
        instance = {}

        sequence = [[bos] + [bg_token] + list(chain(*persona))] + history + [response + ([eos] if self.with_eos else [])]
        sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

        # TODO: whether need to truncate
        truncate_first_turn = False
        new_context_list = self._truncate_list(sequence[:-1], self.max_seq_len - self.max_response_len, truncate_first_turn=truncate_first_turn)
        new_response = sequence[-1]
        if len(new_response) > self.max_response_len:
            new_response = new_response[ : self.max_response_len]
        sequence = new_context_list + [new_response]

        instance["input_ids"] = list(chain(*sequence)) # concatenate input_ids
        # TODO: whether need to add bos for token_type_ids
        instance["token_type_ids"] = [bg_token for _ in sequence[0]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        # instance["token_type_ids"] = [bos] + [persona_token for _ in sequence[0][1:]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] # response + eos + pad
        # instance["lm_labels"] = ([self.pad] * sum(len(s) for s in sequence[:-1])) + [self.pad] + sequence[-1][1:] 

        
        instance["index"] = index
        instance["attention_masks_2d"] = [1]*len(instance["input_ids"])
        
        instance["kg"] = persona # a list

        return instance

    def proprecess_daily(self, index, topic, history, response, tokenizer, task):
        bos, eos, speaker1, speaker2, bg_token = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
        instance = {}
        sequence = [[bos] + [bg_token] + list(topic)] + history + [response + ([eos] if self.with_eos else [])]
        sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

        # TODO: whether need to truncate
        truncate_first_turn = False
        new_context_list = self._truncate_list(sequence[:-1], self.max_seq_len - self.max_response_len, truncate_first_turn=truncate_first_turn)
        new_response = sequence[-1]
        if len(new_response) > self.max_response_len:
            new_response = new_response[ : self.max_response_len]
        sequence = new_context_list + [new_response]

        instance["input_ids"] = list(chain(*sequence)) # concatenate input_ids
        instance["token_type_ids"] = [bg_token for _ in sequence[0]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] # response + eos + pad
        # instance["lm_labels"] = ([self.pad] * sum(len(s) for s in sequence[:-1])) + [self.pad] + sequence[-1][1:] 


        assert len(instance["lm_labels"]) == len(instance["token_type_ids"])
        assert len(instance["token_type_ids"]) == len(instance["input_ids"])
        assert len(instance["input_ids"]) <= self.max_seq_len
        
        instance["index"] = index
        instance["attention_masks_2d"] = [1]*len(instance["input_ids"])

        return instance

    def proprecess_ed(self, index, context, prompt, history, response, tokenizer, task):
        """
        Args:
            prompt: two dim list [[]], the shape should be 1 x N
        """
        bos, eos, speaker1, speaker2, bg_token = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
        instance = {}
        # just simple concat 'context' and 'prompt' together
        # TODO: whether need to drop 'context' key word
        # print("prompt {}".format(prompt) )
        # exit(1)
        sequence = [[bos] + [bg_token] + list(context) + prompt[0]]  +  history + [response + ([eos] if self.with_eos else [])]
        sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

        # truncate
        truncate_first_turn = False
        new_context_list = self._truncate_list(sequence[:-1], self.max_seq_len - self.max_response_len, truncate_first_turn=truncate_first_turn)
        new_response = sequence[-1]
        if len(new_response) > self.max_response_len:
            new_response = new_response[ : self.max_response_len]
        sequence = new_context_list + [new_response]


        instance["input_ids"] = list(chain(*sequence)) # concatenate input_ids
        instance["token_type_ids"] = [bg_token for _ in sequence[0]] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] # response + eos + pad

        # instance["lm_labels"] = ([self.pad] * sum(len(s) for s in sequence[:-1])) + [self.pad] + sequence[-1][1:] 

        
        instance["index"] = index
        instance["attention_masks_2d"] = [1]*len(instance["input_ids"])

        instance["kg"] = prompt # two dim list
    
        return instance

    def proprecess_wow(self, index, topic, persona, history, response, tokenizer, task):
        bos, eos, speaker1, speaker2, bg_token = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task])
        instance = {}
        # TODO: whether need to dorp topci key words
        sequence = [[bos] + [bg_token] + list(topic) + persona[0]] +  history + [response + ([eos] if self.with_eos else [])] # persona one utterance
        sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])]

        # truncate
        truncate_first_turn = False
        new_context_list = self._truncate_list(sequence[:-1], self.max_seq_len - self.max_response_len, truncate_first_turn=truncate_first_turn)
        new_response = sequence[-1]
        if len(new_response) > self.max_response_len:
            new_response = new_response[ : self.max_response_len]
        sequence = new_context_list + [new_response]


        instance["input_ids"] = list(chain(*sequence)) # concatenate input_ids
        # TODO: whether need to add bos for token_type_ids
        instance["token_type_ids"] = [bg_token for _ in sequence[0]]  + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:]) for _ in s]
        # instance["token_type_ids"] = [bos] + [topic_token for _ in sequence[0][1:]] + [persona_token for _ in sequence[1]]  + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[2:]) for _ in s]
        
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:] # response + eos + pad
        
        # instance["lm_labels"] = ([self.pad] * sum(len(s) for s in sequence[:-1])) + [self.pad] + sequence[-1][1:] 

        
        instance["index"] = index
        instance["attention_masks_2d"] = [1]*len(instance["input_ids"])

        instance["kg"] = persona # two dim
    
        return instance
 
    def cal_max_len(self, ids): 
        """ calculate max sequence length """
        if isinstance(ids[0], list): 
            pad_len = max([self.cal_max_len(k) for k in ids])
        else: 
            pad_len = len(ids)
        return pad_len

    def pad_data(self, insts, pad_len, pad_num=-1): 
        """ padding ids """
        insts_pad = []
        if isinstance(insts[0], list): 
            for inst in insts: 
                inst_pad = inst + [self.pad] * (pad_len - len(inst))
                insts_pad.append(inst_pad)
            if len(insts_pad) < pad_num: 
                insts_pad += [[self.pad] * pad_len] * (pad_num - len(insts_pad))
        else: 
            insts_pad = insts + [self.pad] * (pad_len - len(insts))
        return insts_pad
    
    def get_bg_pad(self, batch, key):
        # Add pad horizontally and vertically
        pad_ids = None
        pad_kn_num = None
        pad_kn = max([self.cal_max_len(instance[key]) for instance in batch]) # 不同kn的长度不同，需要pad
        pad_kn_num = max([len(instance[key]) for instance in batch]) # 不同sample的kn数量不同，需要pad
        # print("get bg pad function, pad_kn_num: {}\t pad_kn: {}\nbatch: {}".format(pad_kn_num, pad_kn, batch))
        pad_ids = torch.tensor([self.pad_data(instance[key], pad_kn, pad_kn_num)
                                    for instance in batch], dtype=torch.long)
        
        return pad_ids, pad_kn, pad_kn_num

    def get_memory_mask(self, batch, key, pad_kn_num=None):
        kn_len = [[len(term) for term in instance[key]] for instance in batch]
        kn_len_pad = []
        for elem in kn_len:
            if len(elem) < pad_kn_num:
                elem += [0] * (pad_kn_num - len(elem))
            kn_len_pad.extend(elem)
        kn_len_pad = torch.Tensor(kn_len_pad).long().reshape(-1, pad_kn_num) # [batch, kn_num]
        memory_mask = (kn_len_pad==0)

        return memory_mask

    def collate(self, batch):
        
        has_kg = True 
        for b in batch:
            if "kg" not in b:
                has_kg = False
        # has_kg = False if "kg" not in b for b in batch 

        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        token_type_ids = pad_sequence(
            [torch.tensor(instance["token_type_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        labels = pad_sequence(
            [torch.tensor(instance["lm_labels"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=-1)
        attention_masks_2d = pad_sequence(
            [torch.tensor(instance["attention_masks_2d"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=0)
        indexes = torch.Tensor([torch.tensor(instance["index"], dtype=torch.long) for instance in batch])

        target_ids = torch.Tensor([torch.tensor(instance["target"], dtype=torch.long) for instance in batch]) if not self.lm_labels else None


        kg_pad_ids, kg_memory_mask, kg_pad_kn_num = None, None, None
        if has_kg:
            kg_pad_ids, kg_pad_kn, kg_pad_kn_num = self.get_bg_pad(batch, "kg") # TODO: need to understand
            kg_pad_ids = kg_pad_ids.reshape(-1, kg_pad_kn) # [batch*kg_pad_kn_num, kg_pad_kn]
            # kg_memory_mask = torch.tensor(self.get_memory_mask(batch, "kg", kg_pad_kn_num), dtype=torch.float) # [batch, kg_pad_kn_num]
            kg_memory_mask = self.get_memory_mask(batch, "kg", kg_pad_kn_num).float().clone().detach().requires_grad_(True) # [batch, kg_pad_kn_num]
            kg_pad_kn_num = torch.tensor(kg_pad_kn_num, dtype=torch.long)


        return input_ids, token_type_ids, labels, target_ids, indexes, attention_masks_2d, \
                                                kg_pad_ids, kg_memory_mask, kg_pad_kn_num