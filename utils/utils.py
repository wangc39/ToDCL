import os
import re
import sys
import glob




def make_check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text
        



if __name__ == "__main__":
    pass