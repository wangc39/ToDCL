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
    # len_dialogue = []
    # len_dialogue.append(len([0 for dia in data for t in dia["utterances"]]))
    # print(type(data), data)
    # for index, dia in enumerate(data):
    #     try:
    #         s = len(dia["utterances"][-1]["history"]), len(dia["utterances"][-1]["candidates"])
    #     except TypeError as e:
    #         # print("{}\t{}".format(e, dia["utterances"][-1]["candidates"]))
    #         print(e)
    #         print(index)
    #         # print('dia', dia)
    #         # print('dia["utterances"]', dia["utterances"])
    #         # print(dia["utterances"][-1])
    #         # print("{}\t{}".format(e, dia["utterances"][-1]))

    #         exit(1)
    len_dialogue = [(len(dia["utterances"][-1]["history"]) + len(dia["utterances"][-1]["candidates"])) for dia in data]

    return np.mean(len_dialogue)


def print_sample(data,num):
    color_map = {"USER":"blue","SYSTEM":"magenta","API":"red","API-OUT":"green"}
    for i_d, dial in enumerate(random.sample(data,len(data))):
        print(f'ID:{dial["id"]}')
        # print(f'Services:{dial["services"]}')
        for idx, his in dial["utterances"][0]["history"]:
            print(his)
        for turn in dial['utterances']:
            print(f'{turn["candidates"][0]}')
        if i_d == num: break

def get_datasets(dataset_list=['SGD'], setting="single", verbose=False, develop=False):

    taskConfig = {
        "Cornell": {
            "path": r"./data/CornellMovie/cornell movie-dialogs corpus", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "Ubuntu": {
            "path": r"./data/Ubuntu", 
            "modeList": ["train", "valid", "test"],
            "remake": True,
        },
        "Convai2": {
            "path": r"./data/convai2", 
            "modeList": ["train", "valid"],
            "remake": False,
        },
        "Daily": {
            "path": r"./data/dailydialog", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "Ed": {
            "path": r"./data/empatheticdialogues/visible", 
            "modeList": ["train", "valid", "test"],
            "remake": False,

        },
        "Wow": {
            "path": r"./data/wizard_of_wikipedia", 
            "modeList": ["train", "valid_random_split", "test_random_split"],
            "remake": False,
        },
    }

    preprocessMain = PreprocessMain(taskConfig)

    
    table = []
    train = []
    dev = []
    test = []
    datasets = defaultdict(dict) # "Convai2": [train, dev, test]

    if ("Convai2" in dataset_list):
        print("LOAD Convai2")
        # train_Convai2, dev_Convai2, test_Convai2 = preprocessConvai2(develop=develop)
        train_Convai2, dev_Convai2, test_Convai2 = preprocessMain.process_convai2(develop=develop)
        print("train_Convai2", type(train_Convai2), len(train_Convai2))
        print("---------")

        if(verbose):
            print_sample(train_Convai2,2)
            input()
        # n_domain, n_intent, n_turns, _ = get_domains_slots(train_Convai2)
        n_turns = get_n_turns(train_Convai2)
        table.append({"Name":"Convai2","Trn":len(train_Convai2),"Val":len(dev_Convai2),"Tst":len(test_Convai2), "Tur":n_turns})
        train += train_Convai2
        dev += dev_Convai2
        test += test_Convai2
        datasets["Convai2"] = {"train": train_Convai2, "dev": dev_Convai2, "test": test_Convai2}
        # datasets.append({"Convai2": {"train": train_Convai2, "dev": dev_Convai2, "test": test_Convai2}})


    if ("Ed" in dataset_list):
        print("LOAD ED")
        train_Ed, dev_Ed, test_Ed = preprocessMain.process_ed(develop=develop)

        n_turns = get_n_turns(train_Ed)
        table.append({"Name":"Ed","Trn":len(train_Ed),"Val":len(dev_Ed),"Tst":len(test_Ed), "Tur":n_turns})
        train += train_Ed
        dev += dev_Ed
        test += test_Ed
        datasets["Ed"] = {"train": train_Ed, "dev": dev_Ed, "test": test_Ed}
        # datasets.append({"Ed": {"train": train_Ed, "dev": dev_Ed, "test": test_Ed}})

    if ("Daily" in dataset_list):
        print("LOAD Daily")
        train_Daily, dev_Daily, test_Daily = preprocessMain.process_daily(develop=develop)

        n_turns = get_n_turns(train_Ed)
        table.append({"Name":"Daily","Trn":len(train_Daily),"Val":len(dev_Daily),"Tst":len(test_Daily), "Tur":n_turns})
        train += train_Daily
        dev += dev_Daily
        test += test_Daily
        datasets["Daily"] = {"train": train_Daily, "dev": dev_Daily, "test": test_Daily}
        # datasets.append({"Daily": {"train": train_Daily, "dev": dev_Daily, "test": test_Daily}})

    if ("Cornell" in dataset_list):
        print("LOAD Cornell")
        train_Cornell, dev_Cornell, test_Cornell = preprocessMain.process_cornell(develop=develop)

        n_turns = get_n_turns(train_Cornell)
        table.append({"Name":"Cornell","Trn":len(train_Cornell),"Val":len(dev_Cornell),"Tst":len(test_Cornell), "Tur":n_turns})
        train += train_Cornell
        dev += dev_Cornell
        test += test_Cornell
        datasets["Cornell"] = {"train": train_Cornell, "dev": dev_Cornell, "test": test_Cornell}

    if ("Wow" in dataset_list):
        print("LOAD Wow")
        train_Wow, dev_Wow, test_Wow = preprocessMain.process_wow(develop=develop)

        n_turns = get_n_turns(train_Wow)
        table.append({"Name":"Wow","Trn":len(train_Wow),"Val":len(dev_Wow),"Tst":len(test_Wow), "Tur":n_turns})
        train += train_Wow
        dev += dev_Wow
        test += test_Wow
        datasets["Wow"] = {"train": train_Wow, "dev": dev_Wow, "test": test_Wow}
        # datasets.append({"Wow": {"train": train_Wow, "dev": dev_Wow, "test": test_Wow}})


    if ("Ubuntu" in dataset_list):
        print("LOAD Ubuntu")
        train_Ubuntu, dev_Ubuntu, test_Ubuntu = preprocessMain.process_ubuntu(develop=develop)

        n_turns = get_n_turns(train_Ubuntu)
        table.append({"Name":"Ubuntu","Trn":len(train_Ubuntu),"Val":len(dev_Ubuntu),"Tst":len(test_Ubuntu), "Tur":n_turns})
        train += train_Ubuntu
        dev += dev_Ubuntu
        test += test_Ubuntu
        datasets["Ubuntu"] = {"train": train_Ubuntu, "dev": dev_Ubuntu, "test": test_Ubuntu}

        # datasets.append({"Ubuntu": {"train": train_Ubuntu, "dev": dev_Ubuntu, "test": test_Ubuntu}})

    n_turns = get_n_turns(train)
    table.append({"Name":"TOT","Trn":len(train),"Val":len(dev),"Tst":len(test), "Tur":n_turns})
    print(tabulate(table, headers="keys"))

    return {"TOTAL": {"train":train,"dev":dev,"test":test}, 
            "AllDatasets": datasets}

