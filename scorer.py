import re
import json
from utils.eval_metric import moses_multi_bleu
from utils.f1_metrics import F1Metrics
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
from tabulate import tabulate
import glob
import os.path
from tqdm import tqdm
from dictdiffer import diff
from transformers import GPT2Tokenizer




def evaluate_EER(args,results_dict,entities_json,path, names):
    ERR = []
    cnt_bad = 0
    cnt_superflous = 0
    tot = 0
    
    for d in results_dict:
        if(d["spk"]=="SYSTEM"):
            ent = set()
            ent_corr = []
            if args.task_type == "E2E":
                d['hist'] = d['hist'].split("API-OUT: ")[1]
                if(d['hist']==""):
                    continue

            for speech_act, slot_value_dict in parse_API(d['hist']+" ").items():
                tot += len(slot_value_dict.keys())
                for s,v in slot_value_dict.items():
                    if(v not in ["True", "False", "yes", "no", "?","none"]):
                        if(v.lower() not in d["genr"].lower()):
                            cnt_bad += 1
                        else:
                            ent_corr.append(v.lower())
                        ent.add(v.lower())
                

    return (cnt_bad+cnt_superflous)/float(tot)



def compute_bleu(hyp_text, ref_text):

    # print(type(hyp_text), len(hyp_text), len(hyp_text[0]), len(ref_text))
    # print(hyp_text[0])
    # print(ref_text[0])

    for index, line in enumerate(hyp_text):
        if len(line) != len(hyp_text[0]):
            print(index, len(line), line)


    # hyp_text_array = np.array(hyp_text)
    # print(hyp_text_array[:, 2])
    # print(hyp_text_array)

    BLEU_list = []
    print('Calculating BLEU...')

    for i in range(len(hyp_text[0])):
        
        # print(type(np.array(hyp_text_array[:, i])), np.array(hyp_text[:, i]).shape)
        hyp_text_column = [t[i]for t in hyp_text]
        BLEU = moses_multi_bleu(hyp_text_column, np.array(ref_text))
        BLEU_list.append(BLEU)

    return np.mean(np.array(BLEU_list))

def compute_f1(hyp_word_list, ref_word_list):
    f1_scorer = F1Metrics()
    print('Calculating F1...')
    # f1 = f1_scorer.calculate_metrics([' '.join(item) for item in hyp_word_list], [' '.join(item[0]) for item in ref_word_list])
    avg_f1, max_f1 = f1_scorer.calculate_metrics(hyp_word_list, ref_word_list)
    # task_metrics_dict[cur_test_task_id][cur_train_task_id]["F1-A"] = round(avg_f1*100, 2)
    # task_metrics_dict[cur_test_task_id][cur_train_task_id]["F1-M"] = round(max_f1*100, 2)

    return round(avg_f1*100, 2)


def load_data(hyp_file_path, ref_file_path, tokenizer, nltk_choose=False):
    hyp_word_list = []
    ref_word_list = []

    
    # hyp_word_list:  [[['I', 'love', 'food'], ['Do', 'you', 'like', '?']], [['my', 'name']]] # use punctuation as a word
    with open(hyp_file_path, 'r') as f:
        hyp_data = f.read()
    hyp_words = [line.split("|||") for line in hyp_data.split("\n") ]


    if nltk_choose:
        # use nltk to get word tokens
        hyp_word_list = [[nltk.word_tokenize(s) for s in line] for line in hyp_words]
    else:
        # gpt2 tokenizer
        # "Using a Transformer network is simple." --> ['Using', 'Ġa', 'ĠTrans', 'former', 'Ġnetwork', 'Ġis', 'Ġsimple', '.']
        hyp_word_list = [[tokenizer.tokenize(s) for s in line] for line in hyp_words]

    # ref_word_list:   [[['I', 'love', 'food']], [['Do', 'you', 'like', '?']]] # use punctuation as a word
    with open(ref_file_path, 'r') as f:
        ref_data = f.read()

    ref_words = [line for line in ref_data.split("\n")]

    if nltk_choose:
        ref_word_list = [[nltk.word_tokenize(line)] for line in ref_words] # inorder to keep same format
    else:
        ref_word_list = [[tokenizer.tokenize(line)] for line in ref_words] # inorder to keep same format


    return hyp_word_list, ref_word_list, hyp_words, ref_words


def evaluate(hyp_file_path, ref_file_path, tokenizer, nltk_choose=False):

    hyp_word_list, ref_word_list, hyp_word, ref_word = load_data(hyp_file_path, ref_file_path, tokenizer, nltk_choose=False)

    bleu = compute_bleu(hyp_word, ref_word)
    f1_score = compute_f1(hyp_word_list, ref_word_list)

    return bleu, f1_score



def score_folder():

    parser = ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path to the folder with the results")
    parser.add_argument("--nltk", type=bool, default=False, help="Path to the folder with the results")

    
    args = parser.parse_args()
    # args.task_type = "NLG"
    folders = glob.glob(f"{args.model_checkpoint}/*") # 获取 model_checkpoint 的文件夹的名称


    RESULT = []
    TASK = []
    for folder in folders:
        if "png" in folder or "TOO_HIGH_LR" in folder or "TEMP" in folder or "REPLAY" in folder:
            continue

        tokenizer = GPT2Tokenizer.from_pretrained(folder, bos_token="[bos]", eos_token="[eos]", sos_token="[SOS]", sep_token="[sep]",pad_token='[PAD]')
        
        if "MULTI" in folder:
            pass
            break
        else:
            for index, task_name in enumerate(os.listdir(folder)):
                # taskDict = {}
                hyp_file_path = f'{folder}/{task_name}/result.txt'
                ref_file_path = f'{folder}/{task_name}/gt.txt'
                BLEU, F1 = evaluate(hyp_file_path, ref_file_path, tokenizer, nltk_choose=args.nltk)#, ent=entities_json)
                
                pass
            

        # tokenizer = tokenizer.from_pretrained(args.saving_dir)

        BLEU, F1 = evaluate(hyp_file_path, ref_file_path, tokenizer, nltk_choose=args.nltk)#, ent=entities_json)


        RESULT.append({"Name":folder.split("/")[-1].split("_")[0],"BLEU":BLEU,"F1":F1})

    print(tabulate(RESULT, headers="keys",tablefmt="github"))


score_folder()
