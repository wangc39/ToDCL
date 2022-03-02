
import os
import json
import torch
import shutil
import numpy
import logging
import random
import copy
from tqdm import tqdm
from torch import Tensor
from torch.nn import functional as F
from collections import defaultdict
from argparse import ArgumentParser

# from utils.dataloader import make_loader
from utils.dataloader import get_data_loaders, get_current_task_data, make_loader
from CL_learner import Seq2SeqToD



from utils.dataset_ms import SPECIAL_TOKENS


def _top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits[0].size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value

    return logits


def get_example_inputs(model,tokenizer,prompt_text,device):
    num_attention_heads = model.config.n_head
    hidden_size = model.config.n_embd
    num_layer = model.config.n_layer
    tokenizer.padding_side = "left"
    encodings_dict = tokenizer.batch_encode_plus(prompt_text, padding=True)

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    #Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))

    return input_ids.to(device), attention_mask.to(device), position_ids.to(device), empty_past


def test_generation_GPT2BATCH(model, tokenizer, input_ids, token_type_ids, target_ids, device, current_output=None, do_sample=False, \
                            temperature=1.0,  top_k=0, top_p=0, max_length=30, responses_generate_times=5, repetition_penalty=1.0, task_id=-1):
    
    
    task_name = list(SPECIAL_TOKENS.keys())[0] if task_id == -1 else SPECIAL_TOKENS[task_id]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[task_name])
    bos, eos, speaker1, speaker2, bg_token = special_tokens_ids

    if current_output is None:
        current_output = []
    finish_set = set()

    # repeat the tensor
    input_ids = [copy.deepcopy(input_ids.squeeze(0).numpy().tolist()) for _ in range(responses_generate_times)]
    token_type_ids = [copy.deepcopy(token_type_ids.squeeze(0).numpy().tolist()) for _ in range(responses_generate_times)]
    target_ids = [copy.deepcopy(target_ids.squeeze(0).numpy().tolist()) for _ in range(responses_generate_times)]

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)
    target_ids = torch.tensor(target_ids, dtype=torch.long, device=device)

    # this way, the storage space and gradient space is shared
    # input_ids = input_ids.repeat(responses_generate_times, 1).to(device) # repeat the tensor
    # token_type_ids = token_type_ids.repeat(responses_generate_times, 1).to(device) # repeat the tensor
    # target_ids = target_ids.repeat(responses_generate_times, 1).to(device) # repeat the tensor

    # print("input_ids.shape", input_ids.shape)
    # print("token_type_ids.shape", token_type_ids.shape)
    # print("target_ids.shape", target_ids.shape)


    for i in range(max_length):

        if task_id == -1:
            outputs = model(input_ids, token_type_ids=token_type_ids)
        else:
            outputs = model(input_ids,  token_type_ids=token_type_ids, task_id=task_id)

        logits = outputs[0][:, -1, :] # response logit

        for index in range(responses_generate_times):
            for token_id in set([token_ids[index] for token_ids in current_output]):
                logits[index][token_id] /= repetition_penalty # repeat punishment default equal to 1.0    
        
        
        logits = logits / (temperature if temperature > 0 else 1.0)
        logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p) # batch size x vocab size
        probs = F.softmax(logits, dim=-1)

        # return index of probs according to probs    shape: batch size x 1(5 x 1)
        prev = torch.topk(probs, 1)[1] if do_sample else torch.multinomial(probs, 1) 
        
        for index, token_id in enumerate(prev[:, 0]):
            if token_id == eos:
                finish_set.add(index)
        
        finish_flag = True
        # 如果有一个没有生成完毕的话 继续生成文本
        for index in range(responses_generate_times):
            if index not in finish_set:
                finish_flag = False
                break
        if finish_flag:
            break
        current_output.append([token.item() for token in prev[:, 0]])
        input_ids = torch.cat((input_ids, prev), dim=-1)
        token_type_ids = torch.cat((token_type_ids, token_type_ids[:, -1].unsqueeze(-1)), dim=-1) # just repeat last row

    candidate_responses = []
    for batch_index in range(responses_generate_times):
        response = []
        for token_index in range(len(current_output)):
            # 因为之前选择是 有一个没有生成完毕继续生成文本 所以这里要对之前生成完毕的文本进行筛选 选出已经停止生成的文本
            if current_output[token_index][batch_index] != eos:
                response.append(current_output[token_index][batch_index])
            else:
                break
        candidate_responses.append(response)

    return candidate_responses



def generate_sample_prev_task(args,model,tokenizer,dataset_dic,task_id_so_far,number_of_sample,time,task_id_adpt=-1):
    # device = torch.device(f"cuda:{args.GPU[0]}")
    device = torch.device(f"cuda:0")
    model.to(device)
    model.eval()
    ## notice that this sample is just to have the data struct
    temp_aug_mem = random.sample(dataset_dic["['sgd_restaurants']"],min(len(dataset_dic["['sgd_restaurants']"]),number_of_sample))
    temp_aug_sam = random.sample(dataset_dic["['sgd_restaurants']"],min(len(dataset_dic["['sgd_restaurants']"]),number_of_sample))
    with torch.no_grad():
        if "gpt2" in args.model_checkpoint: ## this works only with GPT2
            sample_list = []
            for i in range(int(number_of_sample/(args.valid_batch_size))+1):
                if(i%2==0 or args.task_type!="E2E"): # sample on batch with and one without API call
                    input_batch = [f"[{str(eval(task_id_so_far)[0])}]" for _ in range(args.valid_batch_size)]
                else:
                    input_batch = [f"[{str(eval(task_id_so_far)[0])}-API]" for _ in range(args.valid_batch_size)]
                _, samples = test_generation_GPT2BATCH(model=model,
                                                    tokenizer=tokenizer,
                                                    input_text=input_batch,
                                                    device=device,
                                                    max_length=300,
                                                    do_sample=True,
                                                    top_p=0.9,
                                                    temperature=1.1,
                                                    task_id=task_id_adpt)
                sample_list += samples
    sample_list = random.sample(sample_list,min(len(sample_list),number_of_sample))
    # this sample is to train the previous task generator
    for i in range(len(temp_aug_mem)):
        temp_aug_mem[i]["history_reply"] = f"{sample_list[i].strip()} {tokenizer.eos_token}"
    # this sample is to train the previous task itself
    # hence we remove the special token in input
    for i in range(len(temp_aug_sam)):
        samp = sample_list[i].strip()
        samp = samp.replace(f"[{str(eval(task_id_so_far)[0])}]","")
        samp = samp.replace(f"[{str(eval(task_id_so_far)[0])}-API]","")
        temp_aug_sam[i]["history_reply"] = f"{samp} {tokenizer.eos_token}"

    temp_aug = temp_aug_mem + temp_aug_sam
    ## save the generated data for logging
    if not os.path.exists(f'{args.saving_dir}/{time}'):
        os.makedirs(f'{args.saving_dir}/{time}')
    with open(f'{args.saving_dir}/{time}'+f'/{task_id_so_far}_generated.json', 'w') as fp:
        json.dump(temp_aug, fp, indent=4)
    return temp_aug

def test_model_seq2seq(args, model, tokenizer, test_loader, result_path, gt_path, task_id):
    
    # device = torch.device(f"cuda:0")
    # model.to(args.device)
    model.eval()

    gt_replys = []
    predictions = []
    # results = []

    for idx_b, batch in tqdm(enumerate(test_loader),total=len(test_loader)):
        with torch.no_grad():
            input_ids, token_type_ids, labels, target_ids, taskname, indexes, attention_masks_2d, \
                                    kg_pad_ids, kg_memory_mask, kg_pad_kn_num = tuple(input_tensor for input_tensor in batch)
                                    # kg_pad_ids, kg_memory_mask, kg_pad_kn_num = tuple(input_tensor for input_tensor in batch)

            # print("input_ids.shape", input_ids.shape)
            # print("token_type_ids.shape", token_type_ids.shape)
            # print("target_ids.shape", target_ids.shape)

            if "gpt2" in args.model_checkpoint:
                candidate_responses = test_generation_GPT2BATCH(model=model,
                                                    tokenizer=tokenizer,
                                                    input_ids=input_ids,
                                                    token_type_ids=token_type_ids,
                                                    target_ids=target_ids,
                                                    device=device,
                                                    responses_generate_times=args.responses_generate_times,
                                                    task_id=task_id,
                                                    max_length = 50)
            else:
                responses = model.generate(input_ids=input_ids,
                                            attention_mask=attention_masks_2d,
                                            token_type_ids=token_type_ids,
                                            eos_token_id=tokenizer.eos_token_id,
                                            max_length=50)
                value_batch = tokenizer.batch_decode(responses, skip_special_tokens=True)
        


            # reply = tokenizer.decode(target_ids, skip_special_tokens=True)
            reply = convert_ids_to_outtext(tokenizer, target_ids)
            reply = reply[0] if isinstance(reply, list) else reply # if list, preservation the first element
            gt_replys.append(reply)
            candidate_texts = convert_ids_to_outtext(tokenizer, candidate_responses)

            predictions.append('|||'.join(candidate_texts))
    
    with open(result_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join(predictions))
    with open(gt_path, 'w', encoding="UTF-8") as f:
        f.write("\n".join(gt_replys))




def convert_ids_to_outtext(tokenizer, candidate_responses):
    candidate_texts = []
    for response in candidate_responses:
        out_text = tokenizer.decode(response, skip_special_tokens=True)
        out_text_pieces = out_text
        # out_text_pieces = ' '.join(jieba.lcut(''.join(out_text.split())))
        candidate_texts.append(out_text_pieces)
    return candidate_texts

def argmin(a):
    return min(range(len(a)), key=lambda x: a[x])

def test_model_seq2seq_ADAPTER(args,model,tokenizer,test_loader,test_dataset,save_dir,max_seen_task=0):
    # device = torch.device(f"cuda:{args.GPU[0]}")
    device = torch.device(f"cuda:0")
    model.model.to(device)
    model.model.eval()
    results = []

    print(model.task_list_seen, len(model.task_list_seen))
    range_adpt = len(model.task_list_seen)

    perplexity_dict = {f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}': [] for sample in test_dataset}
    for t in range(range_adpt):
        print(f"Task {t}")
        for idx_b, batch in tqdm(enumerate(test_loader),total=len(test_loader)):
            ppl_batch = model.compute_PPL(batch,task_id=t,device=device) ## one value per batch
            for (d_id, t_id, ta_id, ppl) in zip(batch["dial_id"],batch["turn_id"],batch["task_id"],ppl_batch):
                perplexity_dict[f'{d_id}_{t_id}_{ta_id}'].append(ppl)

    # select the task id with the lowest perplexity (loss)
    perplexity_dict_ = {}
    for k,v in perplexity_dict.items():
        if len(v) == range_adpt:
            perplexity_dict_[k] = v
        else: 
            print(k,v)


    perplexity_dict = {k: argmin(v) for k,v in perplexity_dict_.items()}


    ## group by sample by predicted task id
    test_dataset_by_predicted_id = defaultdict(list)
    for sample in test_dataset:
        if (f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}' in perplexity_dict):
            test_dataset_by_predicted_id[perplexity_dict[f'{sample["dial_id"]}_{sample["turn_id"]}_{sample["task_id"]}']].append(sample)

    for k,v in test_dataset_by_predicted_id.items():
        print(f"Task {k}: {len(v)}")

    ## create a dataloader for batch each of this
    test_dataset_by_predicted_id = {k: make_loader(args,v,model.tokenizer) for k,v in test_dataset_by_predicted_id.items()}
    
    for pred_task_id, task_loader in tqdm(test_dataset_by_predicted_id.items(),total=len(test_dataset_by_predicted_id)):
        # print(f"Task Id: {task_id}")
        for idx_b, batch in tqdm(enumerate(task_loader),total=len(task_loader)):
            with torch.no_grad():
                value_batch,_ = test_generation_GPT2BATCH(model=model.model,
                                                    tokenizer=model.tokenizer,
                                                    input_text=[b+"[SOS]" for b in batch['history']],
                                                    device=device,
                                                    max_length=100,
                                                    task_id=pred_task_id)
            for idx, resp in enumerate(value_batch):
                results.append({"id":batch["dial_id"][idx],"turn_id":batch["turn_id"][idx],
                                "dataset":batch["dataset"][idx],"task_id":batch["task_id"][idx],
                                "spk":batch["spk"][idx],"gold":batch["reply"][idx],
                                "genr":resp,"hist":batch["history"][idx],"pred_task_id":pred_task_id})
            
            # if(idx_b==1): break
    if not os.path.exists(f'{args.saving_dir}/{time}'):
        os.makedirs(f'{args.saving_dir}/{time}')
    with open(f'{args.saving_dir}/{time}'+'/generated_responses.json', 'w') as fp:
        json.dump(results, fp, indent=4)
    tokenizer.padding_side = "right"




def get_args():
    parser = ArgumentParser()
    # parser.add_argument('--model_checkpoint', type=str, default="gpt2")
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path to the folder with the results")
    parser.add_argument("--test_file_path", type=str, default="", help="Path to the folder with the results")


    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for test") # test dimension equal to 1
    parser.add_argument("--responses_generate_times", type=int, default=5, help="The number of generated response")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients on several steps")

    parser.add_argument("--dataset_list", type=str, default="Ed,Wow,Daily", help="Path for saving")
    parser.add_argument("--max_history", type=int, default=5, help="max number of turns in the dialogue")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--setting", type=str, default="single", help="Path for saving")
    parser.add_argument("--verbose", action='store_true', help="continual baseline")
    parser.add_argument("--test_every_step", action='store_true', help="continual baseline")
    parser.add_argument("--length", type=int, default=50, help="lenght of the generation")
    parser.add_argument("--debug", action='store_true', help="continual baseline")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers")
    parser.add_argument("--bottleneck_size", type=int, default=100)
    parser.add_argument("--number_of_adpt", type=int, default=40, help="number of adapterss")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--percentage_LAM0L", type=float, default=0.2, help="LAMOL percentage of augmented data used")
    parser.add_argument("--reg", type=float, default=0.01, help="CL regularization term")
    parser.add_argument("--episodic_mem_size", type=int, default=100, help="number of batch/sample put in the episodic memory")
    parser.add_argument('--CL', type=str, default="MULTI")

    parser.add_argument('--seed', default=42, type=int)


    hparams = parser.parse_args()

    if(hparams.CL == "ADAPTER"):
        hparams.saving_dir = f"runs/{hparams.dataset_list}/{hparams.CL}_EPC_{hparams.n_epochs}_LR_{hparams.lr}_BOTL_{hparams.bottleneck_size}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    else:
        hparams.saving_dir = f"runs/{hparams.dataset_list}/{hparams.CL}_EM_{hparams.episodic_mem_size}_LAMOL_{hparams.percentage_LAM0L}_REG_{hparams.reg}_PERM_{hparams.seed}_{hparams.model_checkpoint}"
    if(hparams.CL == "MULTI"): 
        hparams.multi = True
        hparams.continual = False
    else: 
        hparams.multi = False
        hparams.continual = True


    return hparams



def test():
    # pass
    args = get_args()
    model = Seq2SeqToD(args)

    args.test_file_path = r"runs/Ed,Wow,Daily/"
    folders = glob.glob(f"{args.test_file_path}/*") # 获取 model_checkpoint 的文件夹的名称
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataloader
    model = Seq2SeqToD(hparams)
    tokenizer = model.tokenizer

    args.multi, args.continual = False, True
    _, _, test_loader, (_, _, _) = get_data_loaders(args, tokenizer=tokenizer, test=True)
    TASKS = list(test_loader.keys())

    task_metrics_dict = {}
    for folder in folders:
        method_name = folder.split("/")[-1].split("_")[0] # get method name from folder path
        for index, (cur_train_task, tasks) in enumerate(test_loader.items()):

            if method_name == "MULTI":
                cur_train_task = "multi"
            task_metrics_dict[cur_train_task] = {}
            for j, (cur_test_task) in enumerate(TASKS[:(index+1)]):
                print("Train Task: {}. Test Task: {}......".format(cur_train_task, cur_test_task))
                task_metrics_dict[cur_test_task][cur_train_task] = {}

                result_path = f'{folder}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_result.txt'
                gt_path = f'{folder}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_gt.txt'

                bleu, f1_score = evaluate(result_path, gt_path, tokenizer, nltk_choose=False)
                task_metrics_dict[cur_test_task]["BLEU4"] = bleu
                task_metrics_dict[cur_test_task]["F1"] = f1_score

    return task_metrics_dict
            
            

def show_table(task_metrics_dict, table_file, res_dir, TASKS):
    method = res_dir.split('/')[-2]
    save_dir = res_dir.split('/')[-1]

    f=open(table_file, 'a+')
    for metric in METRICS:
        tb = pt.PrettyTable()
        tb.field_names = [method] + TASKS + ["avg"]

        train_last_test_all_values = []
        for cur_test_task_id, cur_test_task in enumerate(TASKS):
            cur_row = []
            cur_row.append(cur_test_task)
            train_all_test_first_values = []
            for cur_train_task_id, cur_train_task in enumerate(TASKS):
                # value or "-"
                if cur_test_task_id>=1 and cur_train_task_id <=1:
                    cur_row.append("-")
                    continue

                cur_value = task_metrics_dict[cur_test_task_id][cur_train_task_id][metric]

                # for avg
                if cur_test_task_id==0:
                    train_all_test_first_values.append(cur_value)
                if cur_train_task_id==len(TASKS)-1:
                    train_last_test_all_values.append(cur_value)

                # value or "-"
                cur_row.append(str(cur_value))
                
            if cur_test_task_id==0:
                cur_row.append(str(np.round(np.mean(train_all_test_first_values), 2)))
            else:
                cur_row.append("-")
            tb.add_row(cur_row)

        cur_row = ["-"]*len(TASKS)
        cur_row.append(str(np.round(np.mean(train_last_test_all_values), 2)))
        cur_row.append("-")
        tb.add_row(cur_row)

        f.write("method: {}, save_dir: {}, metric: {}\n".format(method, save_dir, metric))
        f.write(str(tb))
        f.write("\n")
    f.write("\n")
    f.write("\n")
    f.close()






# def test():
#     # pass
#     args = get_args()
#     model = Seq2SeqToD(args)

#     args.test_file_path = r"runs/Ed,Wow,Daily/"
#     folders = glob.glob(f"{args.test_file_path}/*") # 获取 model_checkpoint 的文件夹的名称


#     args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # load dataloader
#     args.multi, args.continual = False, True
#     _, _, test_loader, (_, _, _) = get_data_loaders(args, tokenizer=model.tokenizer, test=True)
#     TASKS = list(test_loader.keys())

#     task_metrics_dict = {}
#     for folder in folders:
#         method_name = folder.split("/")[-1].split("_")[0] # get method name from folder path
#         if "MULTI" in folder:
#             result_path = f'{folder}/FINAL'+'/multiSkill_result.txt'
#             gt_path = f'{folder}/FINAL'+'/multiSkill_gt.txt'
#             if os.path.exists(result_path) and os.path.exists(gt_path): 
#                 pass
#             else:
#                 args.multi, args.continual = True, False
#                 # load dataloader
#                 _, _, all_test_loader, (_, _, _) = get_data_loaders(args, tokenizer=model.tokenizer, test=True)

#                 # load model and tokenizer
#                 tokenizer_dir = os.path.join(folder, "tokenizer_config.json")
#                 model.tokenizer.from_pretrained(folder)
#                 # load train model
#                 model_dir = os.path.join(folder, "pytorch_model.bin")
#                 model.model.load_state_dict(torch.load(model_dir))
#                 model.model.to(args.device)

#                 test_model_seq2seq(args, model.model, model.tokenizer, test_loader, result_path, gt_path)

#             bleu, f1score = evaluate(hyp_file_path, ref_file_path, model.tokenizer, nltk_choose=False)
#             # task_metrics_dict["multi"][""]
#         else:
#             for index, (cur_train_task, tasks) in enumerate(test_loader.items()):
                
#                 # load train model
#                 cur_path = os.path.join(folder, f"{index}_{cur_train_task}")
#                 tokenizer_dir = os.path.join(cur_path, "tokenizer_config.json")
#                 model.tokenizer.from_pretrained(cur_path)
#                 model_dir = os.path.join(cur_path, "pytorch_model.bin")
#                 model.model.load_state_dict(torch.load(model_dir))
#                 model.model.to(args.device)

#                 print(f"Loading task {cur_train_task} train Model")

#                 for j, (cur_test_task) in enumerate(TASKS[:(index+1)]):

#                     print("Train Task: {}. Test Task: {}......".format(cur_train_task, cur_test_task))

#                     result_path = f'{folder}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_result.txt'
#                     gt_path = f'{folder}/FINAL'+f'/multiSkill_test_{cur_test_task}_train_{cur_train_task}_gt.txt'
                    
#                     if method_name == "ADAPTER":
#                         test_model_seq2seq(args, model.model, model.tokenizer, test_loader[cur_test_task], 
#                                                                     save_dir, result_path, gt_path, task_id)
#                     else:
#                         test_model_seq2seq(args, model.model, model.tokenizer, test_loader[cur_test_task], 
#                                                                     save_dir, result_path, gt_path, task_id)        
            
            




    # test_all_file(args)

if __name__ == "__main__":
    test()
    pass
