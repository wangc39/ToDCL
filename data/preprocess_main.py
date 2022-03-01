import os, json, codecs, csv
import codecs
import random
import string
import copy

import numpy as np


class PreprocessMain:

    def __init__(self, taskConfig, seed=42, history_min_length=2):

        # path
        self.taskConfig = taskConfig

        # setting
        self.history_min_length = history_min_length
        self.punc = string.punctuation
        self.seed = seed

        # variable
        self.length_dict = {}

        # ignore encoding error
        codecs.register_error('strict', codecs.ignore_errors)
        # self.check_folder_exists_make(self.generatePath)

        self.set_seed(self.seed)

    def set_seed(self, seed):
        """Fixes randomness to enable reproducibility.
        """
        if seed is None:
            raise Exception("Seed is None!")
        random.seed(seed)


    def process_cornell(self, develop=False):

        path, modeList = self.taskConfig["Cornell"]["path"], self.taskConfig["Cornell"]["modeList"]


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
            dialogue_id = {"train": 0, "valid": 0, "test": 0}
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
                    instance["dataset"] = "[Cornell]"
                    instance["dialogue_id"] = dialogue_id[mode]
                    dialogue_id[mode] += 1
                    for i in range(self.history_min_length, len(texts)):
                        utterance = {}
                        utterance["utterance_id"] = i - self.history_min_length
                        utterance["history"] = [text.strip() for text in texts[:i]] # :2 == 0 1 
                        utterance["candidates"] = [texts[i].strip()]
                        instance["utterances"].append(utterance) # add dict to list

                    # if instance is not empty, extend instance
                    if instance["utterances"]: 
                        outputs[mode].append(instance)

        train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
        return train_data, dev_data, test_data

    def process_ubuntu(self, develop=False):
        path, modeList = self.taskConfig["Ubuntu"]["path"], self.taskConfig["Ubuntu"]["modeList"]


        outputs = {}
        for mode in modeList:
            outputs[mode] = []
            csvPath = os.path.join(path, "{}.csv".format(mode))
            with codecs.open(csvPath, "r") as csv_file:

                csv_read = csv.reader(csv_file, delimiter=',') # , delimiter=','
                next(csv_read)  # eat header
                for index, line in enumerate(csv_read):
                    
                    instance = {}
                    fields = [
                        s.replace('__eou__', '.').replace('__eot__', '\n').strip()
                        for s in line
                    ]
                    context = fields[0]
                    response = fields[1]

                    instance["utterances"] = []
                    instance["dialogue_id"] = index
                    instance["dataset"] = "[Ubuntu]"
                    texts = context.split("\n") + [response] # concat context and response
                    for i in range(self.history_min_length, len(texts)):
                        utterance = {}
                        utterance["history"] = texts[:i] # :2 == 0 1 
                        utterance["candidates"] = [texts[i]]
                        utterance["utterance_id"] = i - self.history_min_length
                        instance["utterances"].append(utterance) # add dict to list

                    cands = None
                    if len(fields) > 3:
                        cands = [fields[i] for i in range(2, len(fields))]
                        cands.append(response)
                        random.shuffle(cands)
                    if cands:
                        instance["candidates_last"] = cands

                    if instance["utterances"]: 
                        outputs[mode].append(instance)
        
        train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
        return train_data, dev_data, test_data

    def process_convai2(self, develop=False):

        def deal_persona_dialogue(your_persona, partner_persona, dialogues, dialogues_id=0):
            '''return traditional instance
            your_persona: a list ['', '', '', '']
            partner_persona: a list ['', '', '', '']
            dialogues: dialogues list['', '', '', '', '', ...]
            return: instance
            '''
            # if len(your_persona) != 4:
            #     print(your_persona)
            #     raise Exception("len(your_persona) != 4")
            # if len(partner_persona) != 4:
            #     print(partner_persona)
            #     raise Exception("len(partner_persona) != 4")


            instance = {}
            instance["dataset"] = "[Convai]"
            instance["dialogue_id"] = dialogues_id
            instance["speaker1"] = copy.deepcopy(your_persona) 
            instance["speaker2"] = copy.deepcopy(partner_persona)
            instance["utterances"] = []
            for index in range(self.history_min_length, len(dialogues)):
                utterance = {}
                utterance["history"] = [text.strip() for text in dialogues[:index]]
                utterance["candidates"] = [dialogues[index].strip()]
                utterance["utterance_id"] = index - self.history_min_length
                instance["utterances"].append(utterance)
            
            # instance["turns"] = dialogues
            
            return instance

        path, modeList = self.taskConfig["Convai2"]["path"], self.taskConfig["Convai2"]["modeList"]
        
        your_persona_flag = "your persona: "
        partner_persona_flag = "partner's persona: "
        outputs = {}
        for mode in modeList:
            outputs[mode] = []
            filePath = os.path.join(path, "{}_both_original_no_cands.txt".format(mode))

            with codecs.open(filePath, "r") as fp:
    
                speaker1 = []
                speaker2 = []
                dialogues = []
                dialogues_id = 0

                lines = fp.readlines()
                speaker2First = False if lines[0].startswith(your_persona_flag) else True
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

                            instance = deal_persona_dialogue(speaker1, speaker2, dialogues, dialogues_id)
                            dialogues_id += 1
                            if instance["utterances"]: 
                                outputs[mode].append(instance)
                            speaker1.clear()
                            speaker2.clear()
                            dialogues.clear()
                            if(develop and dialogues_id ==10): break


                    else:
                        # in the final
                        instance = deal_persona_dialogue(speaker1, speaker2, dialogues)
                        if instance["utterances"]: 
                            outputs[mode].append(instance)
                        speaker1.clear()
                        speaker2.clear()
                        dialogues.clear()
                    
        train, vaild = outputs["train"], outputs["valid"]
        data = train + vaild

        train_data, dev_data, test_data = np.split(data, [int(len(data)*0.7), int(len(data)*0.8)])
        train_data, dev_data, test_data = train_data.tolist(), dev_data.tolist(), test_data.tolist() # from numpy to list

        return train_data, dev_data, test_data


    def process_daily(self, develop=False):

        path, modeList = self.taskConfig["Daily"]["path"], self.taskConfig["Daily"]["modeList"]

        outputs = {}
        for mode in modeList:
            outputs[mode] = []
            jsonPath = os.path.join(path, "{}.json".format(mode))

            with codecs.open(jsonPath, "r") as fp:
                for index, line in enumerate(fp.readlines()):

                    line = json.loads(line)
                    instance = {}

                    dialog = line["dialogue"]
                    topic = line["topic"]   
                    # TODO: whether need to split according _
                    # topic = (" ").join(topic.split("_"))
                    instance["dataset"] = "[Daily]"
                    instance["dialogue_id"] = index
                    instance["topic"] = topic
                    instance["utterances"] = []
                    for i in range(self.history_min_length, len(dialog)):
                        
                        utterance = {}
                        utterance["candidates"] = [dialog[i]["text"].strip()] # only need last utterance
                        utterance["history"] = [item["text"].strip() for item in dialog[:i][:]]
                        utterance["utterance_id"] = i - self.history_min_length

                        instance["utterances"].append(utterance)
                    
                    if instance["utterances"]:
                        outputs[mode].extend([instance])

        train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
        return train_data, dev_data, test_data


    def process_ed(self, develop=False):

        path, modeList = self.taskConfig["Ed"]["path"], self.taskConfig["Ed"]["modeList"]

        outputs = {}
        for mode in modeList:
            outputs[mode] = []
            jsonPath = os.path.join(path, "{}.json".format(mode))
            # print(os.getcwd())
            with codecs.open(jsonPath, "r") as fp:
                data = json.load(fp)


            for index, line in enumerate(data):
                instance = {}
                dialog = line["dialog"]
                instance["dataset"] = "[Ed]"
                instance["dialogue_id"] = index
                instance["prompt"] = [line["prompt"].strip()]
                instance["context"] = line["context"].strip()
                instance["utterances"] = []
                for i in range(self.history_min_length, len(dialog)):
                    utterance = {}
                    utterance["candidates"] = [dialog[i]["utterance"].strip()]
                    utterance["history"] = [item["utterance"].strip() for item in dialog[:i][:]]
                    utterance["utterance_id"] = i - self.history_min_length
                    instance["utterances"].append(utterance)
                
                if instance["utterances"]:
                    outputs[mode].extend([instance])

        
        train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
        return train_data, dev_data, test_data

    
    def process_wow(self, develop=False):
        '''
        # TODO: wheather need to add the "retrieved_passages" item to context or not. (may do not need)
        mode: train/valid_random_split/test_random_split
        output: list[list[str]]
        '''

        path, modeList = self.taskConfig["Wow"]["path"], self.taskConfig["Wow"]["modeList"]


        outputs = {"train": [], "valid": [], "test": []}
        for mode in modeList:

            input_f = os.path.join(path, "{}.json".format(mode))
            mode_split = mode.split("_")[0] if "_" in mode else mode 

            with open(input_f, "r") as f:
                data = json.load(f)

            for index, d in enumerate(data):

                instance = {}
                dialog = d["dialog"]
                # "speaker": "0_Wizard"
                if "wizard" in dialog[0]["speaker"].lower():
                    start = (self.history_min_length + 1) if self.history_min_length % 2 else self.history_min_length
                # "speaker": "0_Apprentice"
                elif "apprentice" in dialog[0]["speaker"].lower():
                    start = self.history_min_length if self.history_min_length % 2 else (self.history_min_length + 1)
                else:
                    raise Exception("Wow dataset do not begin with wizard or apprentice")

                instance["dataset"] = "[Wow]"
                instance["dialogue_id"] = index
                instance["topic"] = d["chosen_topic"]
                instance["persona"] = [d["persona"].strip()]
                instance["utterances"] = []
                for i in range(start, len(dialog)):
                    utterance = {}
                    utterance["utterance_id"] = i - start
                    utterance["candidates"] = [dialog[i]["text"].strip()]
                    utterance["history"] = [item["text"].strip() for item in dialog[:i][:]]
                    instance["utterances"].append(utterance)

                if instance["utterances"]: 
                    outputs[mode_split].extend([instance])

        train_data, dev_data, test_data = outputs["train"], outputs["valid"], outputs["test"]
        return train_data, dev_data, test_data


    def save(self, taskname, outputs):
        with codecs.open(os.path.join(self.generatePath, "multiSkill_{}.json".format(taskname)), "w") as f:
            json.dump(outputs, f, indent=4)
    
    def check_file_exists(self, taskname):
        return os.path.isfile(os.path.join(self.generatePath, "multiSkill_{}.json".format(taskname))) 

    def check_folder_exists_make(self, path):
        if not os.path.exists(path):
            os.makedir(path)


    def run(self):
        for index, (taskname, taskConfig) in enumerate(self.path.items()):
            
            # wether need to regenerate
            remake = taskConfig["remake"]

            if self.check_file_exists(taskname): 
                print("Task {} has already dealed".format(taskname))
                if not remake: 
                    continue 
                print("Task {} has remaked".format(taskname))
                

            print("Begin to deal task {}".format(taskname))
            self.length_dict[taskname] = {}
            if taskname == "cornell":
                outputs = self.process_cornell(taskConfig)
            elif taskname == "ubuntu":
                outputs = self.process_ubuntu(taskConfig)           
            elif taskname == "convai2":
                outputs = self.process_convai2(taskConfig)
            elif taskname == "ed":
                outputs = self.process_ed(taskConfig)
            elif taskname == "daily":
                outputs = self.process_daily(taskConfig)
            elif taskname == "wow":
                outputs = self.process_wow(taskConfig)
            else:
                raise Exception("Can not find task {}".format(taskname))       
            
            if outputs:
                self.save(taskname, outputs)
                print("Finish deal task {}".format(taskname))
            else:
                raise Exception("process function return none")


    def show_information(self):
        '''in order to show the information of dealed dataset
        '''
        print("Begin to show information of datasets")
        information_dict = {}
        for index, (taskname, taskConfig) in enumerate(self.path.items()):

            path, modeList = taskConfig["path"], taskConfig["modeList"]
            information_dict[taskname] = {}
            filePath = os.path.join(self.generatePath, "multiSkill_{}.json".format(taskname))

            print("task {}".format(taskname))


            with codecs.open(filePath, "r") as fp:
                data = json.load(fp)

            modeDataLength, responseLength = 0, 0
            responseTurnLength, turnCount, responseCount= 0, 0, 0 

            for mode in modeList:
                mode = mode.split("_")[0] if "_" in mode else mode  # _ in mode
                modeData = data[mode]
                modeDataLength += len(modeData)
                for index, item in enumerate(modeData):
                    firstTurn = item["utterances"][0]
                    lastTurn = item["utterances"][-1] # last
                    totalTurns = lastTurn["history"] + lastTurn["candidates"] # total utterances

                    turnCount += len(item["utterances"])
                    responseCount += len(totalTurns) - len(firstTurn["history"]) + 1
                    # assert len(item["utterances"]) == 
                    for turnIndex in range(len(firstTurn["history"]), len(totalTurns)):
                        # cout the number the of words; if not split, count characters
                        responseTurnLength += len(totalTurns[turnIndex].split())
                
                print("{} {}".format(mode, len(modeData)), end="\t")

            responseLength = responseTurnLength / responseCount
            turnLength  = turnCount / modeDataLength
            print("\nTurns {}\t response length {}".format(turnLength, responseLength))
            print("====="*20)




if __name__ == "__main__":



    path = {
        "Cornell": {
            "path": r"../data/CornellMovie/cornell movie-dialogs corpus", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "Ubuntu": {
            "path": r"../data/Ubuntu", 
            "modeList": ["train", "valid", "test"],
            "remake": True,
        },
        "Convai2": {
            "path": r"../data/convai2", 
            "modeList": ["train", "valid"],
            "remake": False,
        },
        "Daily": {
            "path": r"../data/dailydialog", 
            "modeList": ["train", "valid", "test"],
            "remake": False,
        },
        "Ed": {
            "path": r"../data/empatheticdialogues/visible", 
            "modeList": ["train", "valid", "test"],
            # "remake": True,
            "remake": False,

        },
        "Wow": {
            "path": r"../data/wizard_of_wikipedia", 
            "modeList": ["train", "valid_random_split", "test_random_split"],
            "remake": False,
        },
    }
    generatePath = r"../data/multiSkill_dataset_v2/"
    p = PreprocessMain(path, generatePath)
    p.run()
    p.show_information()
