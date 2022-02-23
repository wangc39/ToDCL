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
from data.preprocess_main import PreprocessMain

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

    path = {
        "cornell": {
            "path": r"CornellMovie/cornell movie-dialogs corpus", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "ubuntu": {
            "path": r"Ubuntu", 
            "modeList": ["train", "valid", "test"],
            "remake": True,
        },
        "convai2": {
            "path": r"convai2", 
            "modeList": ["train", "valid"],
            "remake": False,
        },
        "daily": {
            "path": r"dailydialog", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "ed": {
            "path": r"empatheticdialogues/visible", 
            "modeList": ["train", "valid", "test"],
            # "remake": True,
            "remake": False,

        },
        "wow": {
            "path": r"wizard_of_wikipedia", 
            "modeList": ["train", "valid_random_split", "test_random_split"],
            "remake": False,
        },
    }
    # generatePath = r"../data/multiSkill_dataset_v2/"
    preprocessMain = PreprocessMain(path, generatePath)
    # p.run()
    # p.show_information()
    
    table = []
    train = []
    dev = []
    test = []
    datasets = [] # "Convai2": [train, dev, test]

    if ("Convai2" in dataset_list):
        print("LOAD Convai2")
        # train_Convai2, dev_Convai2, test_Convai2 = preprocessConvai2(develop=develop)
        train_Convai2, dev_Convai2, test_Convai2 = preprocessMain.process_convai2(develop=develop)

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
        train_Ed, dev_Ed, test_Ed = preprocessMain.process_ed(develop=develop)

        n_turns = get_n_turns(train_Ed)
        table.append({"Name":"Ed","Trn":len(train_Ed),"Val":len(dev_Ed),"Tst":len(test_Ed), "Tur":n_turns})
        train += train_Ed
        dev += dev_Ed
        test += test_Ed
        datasets.append({"Ed": {"train": train_Ed, "dev": dev_Ed, "test": test_Ed}})

    if ("Daily" in dataset_list):
        print("LOAD Daily")
        train_Daily, dev_Daily, test_Daily = preprocessMain.process_daily(develop=develop)

        n_turns = get_n_turns(train_Ed)
        table.append({"Name":"Daily","Trn":len(train_Daily),"Val":len(dev_Daily),"Tst":len(test_Daily), "Tur":n_turns})
        train += train_Daily
        dev += dev_Daily
        test += test_Daily
        datasets.append({"Daily": {"train": train_Daily, "dev": dev_Daily, "test": test_Daily}})


    n_turns = get_n_turns(train)
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test), "Tur":n_turns})
    print(tabulate(table, headers="keys"))

    return {"TOTAL": {"train":train,"dev":dev,"test":test}, 
            "AllDatasets": datasets}

