import glob
import json
import csv
import re
import codecs
import numpy as np
from tqdm import tqdm
from termcolor import colored
from collections import defaultdict
from tabulate import tabulate

history_min_length = 2

def preprocessDaily_(split, develop=False):
    path = "dailydialog"
    outputs = []

    jsonPath = os.path.join(path, "{}.json".format(split))

    with codecs.open(jsonPath, "r") as fp:
        data = json.load(fp)

    for index, line in enumerate(data):
        instance = {}
        dialog = line["dialog"]
        instance["id"] = index
        instance["datasets"] = "Daily"
        instance["prompt"] = [line["prompt"].strip()]
        instance["context"] = line["context"].strip()
        instance["utterances"] = []
        for i in range(history_min_length, len(dialog)):
            utterance = {}
            utterance["candidates"] = [dialog[i]["utterance"].strip()]
            utterance["history"] = [item["utterance"].strip() for item in dialog[:i][:]]
            
            instance["utterances"].append(utterance)
        
        if instance["utterances"]:
            outputs.extend([instance])

    return outputs



def preprocessDaily(develop=False):

    train_data = preprocessDaily_("train", develop=False)
    dev_data = preprocessConvai2_("valid", develop=False)
    test_data = preprocessConvai2_("test", develop=False)

    # train_data, dev_data, test_data = np.split(data, [int(len(data)*0.7), int(len(data)*0.8)])
    return train_data, dev_data, test_data



