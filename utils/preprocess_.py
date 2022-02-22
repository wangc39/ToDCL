from data.MWOZ import preprocessMWOZ
from data.SGD import preprocessSGD
from data.TM import preprocessTM2019,preprocessTM2020
from data.Convai2 import preprocessConvai2
from termcolor import colored
import numpy as np
import random
import re
from tabulate import tabulate
from termcolor import colored
from collections import defaultdict

def get_n_turns(data):
    len_dialogue = []
    len_dialogue.append(len([0 for dia in data for t in dia["dialogue"] ]))
    return np.mean(len_dialogue)


def print_sample(data,num):
    color_map = {"USER":"blue","SYSTEM":"magenta","API":"red","API-OUT":"green"}
    for i_d, dial in enumerate(random.sample(data,len(data))):
        print(f'ID:{dial["id"]}')
        print(f'Services:{dial["services"]}')
        for turn in dial['dialogue']:
            print(colored(f'{turn["spk"]}:',color_map[turn["spk"]])+f' {turn["utt"]}')
        if i_d == num: break

def get_datasets(dataset_list=['SGD'],setting="single",verbose=False,develop=False):

    table = []
    train = []
    dev = []
    test = []
    datasets = [] # "Convai2": [train, dev, test]

    if ("Convai2" in dataset_list):
        print("LOAD Convai2")
        train_Convai2, dev_Convai2, test_Convai2 = preprocessConvai2(develop=develop)
        if(verbose):
            print_sample(train_Convai2,2)
            input()
        # n_domain, n_intent, n_turns, _ = get_domains_slots(train_Convai2)
        n_turns = get_n_turns(train_Convai2)
        table.append({"Name":"Convai2","Trn":len(train_Convai2),"Val":len(dev_Convai2),"Tst":len(test_Convai2), "Tur":n_turns})
        train += train_Convai2
        dev += dev_Convai2
        test += test_Convai2
        datasets.append({"Convai2": {"train": train_Convai2, "dev": dev_Convai2, "test": test_Convai2}})


    if ("Ed" in dataset_list):
        print("LOAD ED")
        train_Ed, dev_Ed, test_Ed = preprocessEd(develop=develop)
        n_turns = get_n_turns(train_Ed)
        table.append({"Name":"TM19","Trn":len(train_Ed),"Val":len(dev_Ed),"Tst":len(test_Ed), "Tur":n_turns})
        train += train_Ed
        dev += dev_Ed
        test += test_Ed
        datasets.append({"Ed": {"train": train_Ed, "dev": dev_Ed, "test": test_Ed}})



    n_turns = get_n_turns(train)
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test), "Tur":n_turns})
    print(tabulate(table, headers="keys"))

    return {"TOTAL": {"train":train,"dev":dev,"test":test}, 
            "AllDatasets": datasets}

