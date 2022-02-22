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

def preprocessCornell(develop=False):

    path, modeList = "CornellMovie/cornell movie-dialogs corpus", ["train", "valid", "test"]

    lines_files = os.path.join(path, "movie_lines.txt")
    convo_file = os.path.join(path, "movie_conversations.txt")

    def get_lines(lines_files):
        lines = {}
        # get conversation from lines files
        with codecs.open(lines_files, "r") as fp:
            for line in fp.readlines():
                l = line.split(" +++$+++ ")
                lines[l[0]] = ' '.join(l[4:]).strip('\n').replace('\t', ' ')
        return lines


    lines = get_lines(lines_files)

    cnt = 0
    outputs = {}
    for mode in modeList:
        outputs[mode] = []
        with codecs.open(convo_file, "r") as fp:
            for cnt, line in enumerate(fp.readlines(), 1):
                l = line.split(' ')
                convo = ' '.join(l[6:]).strip('\n').strip('[').strip(']')
                c = convo.replace("'", '').replace(' ', '').split(',')
                texts = [lines[l] for l in c] # get a round of dialogue

                # print(cnt)
                if (cnt % 10 == 0) and mode != 'test': continue
                elif (cnt % 10 == 1) and mode != 'valid': continue
                elif (cnt % 10 > 1) and mode != 'train': continue


                instance = {}
                # get history and response
                instance["utterances"] = []
                for i in range(history_min_length, len(texts)):
                    utterance = {}
                    utterance["history"] = [text.strip() for text in texts[:i]] # :2 == 0 1 
                    utterance["candidates"] = [texts[i].strip()]
                    instance["utterances"].append(utterance) # add dict to list

                # if instance is not empty, extend instance
                if instance["utterances"]: 
                    outputs[mode].append(instance)
    train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
    return train_data, dev_data, test_data


