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


def preprocessConvai2_(split, develop=False):
    def deal_persona_dialogue(your_persona, partner_persona, dialogues):
        '''return traditional instance
        your_persona: a list ['', '', '', '']
        partner_persona: a list ['', '', '', '']
        dialogues: dialogues list['', '', '', '', '', ...]
        return: instance
        '''

        instance = {}
        instance["speaker1"] = copy.deepcopy(your_persona) 
        instance["speaker2"] = copy.deepcopy(partner_persona)
        turns = []
        for index in range(self.history_min_length, len(dialogues)):
            turns.append({"dataset":"Convai2","id": dialogue_id,"turn_id": index, 
                            "spk": "SYSTEM" if index % 2 else "USER","utt": dialogues[index].strip()})

        instance["turns"] = turns
        return instance

    data = []
    dialogue = json.load(open(f"data/convai2/{split}_both_original_no_cands.txt.json"))
    with codecs.open(filePath, "r") as fp:    
        lines = fp.readlines()   

    speaker1 = []
    speaker2 = []
    dialogues = []

    your_persona_flag = "your persona: "
    partner_persona_flag = "partner's persona: "
    speaker2First = False if lines[0].startswith(your_persona_flag) else True
    dialogue_id = 0
    for index, line in enumerate(lines):  
        line = line.split(" ", 1)[-1].strip() # split one time
        if index != len(lines) - 1:
            nextSerial, nextLine = lines[index+1].split(" ", 1) # split one time
            nextLine = nextLine.strip()

            ##### reserve the "__SILENCE__" information
            # line = line.replace("__SILENCE__", "").strip()
            # if line[0] in self.punc: line = line[1:].strip() # delete the punc in the beginning
            
            # add to speaker1 and speaker2 list
            if not speaker2First:
                if line.startswith(your_persona_flag):
                    speaker1.append(line.split(your_persona_flag)[-1].strip())
                elif line.startswith(partner_persona_flag):
                    speaker2.append(line.split(partner_persona_flag)[-1].strip())
            else:
                if line.startswith(your_persona_flag):
                    speaker2.append(line.split(your_persona_flag)[-1].strip())
                elif line.startswith(partner_persona_flag):
                    speaker1.append(line.split(partner_persona_flag)[-1].strip()) 
                
            # don't begin with persona, add to dialogues list
            if not line.startswith(your_persona_flag) and not line.startswith(partner_persona_flag):
            # if your_persona_flag and partner_persona_flag not in line:
                dialogue = [d.strip() for d in line.split("\t")]
                # assert len(dialogue) == 2
                if len(dialogue) != 2:
                    print(line, type(line))
                    raise Exception("dialogue: {}".format(dialogue))

                dialogues.extend(dialogue)
            
            # deal with
            if len(dialogues) and int(nextSerial) == 1:
                # judge the next first speaker
                speaker2First = True if nextLine.startswith(partner_persona_flag) else False

                instance = deal_persona_dialogue(speaker1, speaker2, dialogues)
                if instance["utterances"]: 
                    outputs[mode].append(instance)
                speaker1.clear()
                speaker2.clear()
                dialogues.clear()

        else:
            # in the final
            instance = deal_persona_dialogue(speaker1, speaker2, dialogues)
            dia = {"id": dialogue_id, "dataset": "Convai2", 
                    "kg": " ".join(instance["speaker1"] + instance["speaker2"]), "turns": instance["turns"]}
            dialogue_id += 1
            data.append(dia)
            if instance["utterances"]: 
                outputs[mode].append(instance)
            speaker1.clear()
            speaker2.clear()
            dialogues.clear()
        if(develop and dialogue_id == 1): break

    return data



def preprocessUbuntu(develop=False):

    data = preprocessConvai2_("train", develop=False)
    data += preprocessConvai2_("valid", develop=False)
    train_data, dev_data, test_data = np.split(data, [int(len(data)*0.7), int(len(data)*0.8)])
    return train_data, dev_data, test_data



