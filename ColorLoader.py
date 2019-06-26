from __future__ import print_function

import os
import json
import torch 
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import colorsys
# from src.utils.utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, 'data')
print(RAW_DIR)

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
def process_texts(self, texts):
    inputs, targets, lengths, positions = [], [], [], []

    n = len(texts)
    for text in texts:
        text = text.lower()

    max_len = 0
    for i in range(n):
        # remove punc
        # all lowercase
        text = texts[i].lower()
        # text = texts[i].translate(str.maketrans('', '', string.punctuation))

        tokens = word_tokenize(text)
        input_tokens = [SOS_TOKEN] + tokens
        target_tokens = tokens + [EOS_TOKEN]
        assert len(input_tokens) == len(target_tokens)
        length = len(input_tokens)
        max_len = max(max_len, length)

        inputs.append(input_tokens)
        targets.append(target_tokens)
        lengths.append(length)

    for i in range(n):
        input_tokens = inputs[i]
        target_tokens = targets[i]
        length = lengths[i]
        input_tokens.extend([PAD_TOKEN] * (max_len - length))
        target_tokens.extend([PAD_TOKEN] * (max_len - length))
        input_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in input_tokens]
        target_tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in target_tokens]
        pos = [pos_i+1 if w_i != self.pad_index else 0
                for pos_i, w_i in enumerate(input_tokens)]
        inputs[i] = input_tokens
        targets[i] = target_tokens
        positions.append(pos)
    
    inputs = np.array(inputs)
    targets = np.array(targets)
    lengths = np.array(lengths)
    positions = np.array(positions)

    return inputs, targets, lengths, positions, max_len
def hsl2rgb(hsl):
    """Convert HSL coordinates to RGB coordinates.
    https://www.rapidtables.com/convert/color/hsl-to-rgb.html
    @param hsl: np.array of size 3
                contains H, S, L coordinates
    @return rgb: (integer, integer, integer)
                 RGB coordinate
    """
    H, S, L = hsl[0], hsl[1], hsl[2]
    assert (0 <= H <= 360) and (0 <= S <= 1) and (0 <= L <= 1)

    C = (1 - abs(2 * L - 1)) * S
    X = C * (1 - abs((H / 60.) % 2 - 1))
    m = L - C / 2.

    if H < 60:
        Rp, Gp, Bp = C, X, 0
    elif H < 120:
        Rp, Gp, Bp = X, C, 0
    elif H < 180:
        Rp, Gp, Bp = 0, C, X
    elif H < 240:
        Rp, Gp, Bp = 0, X, C
    elif H < 300:
        Rp, Gp, Bp = X, 0, C
    elif H < 360:
        Rp, Gp, Bp = C, 0, X

    R = int((Rp + m) * 255.)
    G = int((Gp + m) * 255.)
    B = int((Bp + m) * 255.)
    return (R, G, B)

def load_item_id_to_utterance_map():
    with open(os.path.join(RAW_DIR, 'filteredCorpus.csv')) as fp:
        df = pd.read_csv(fp)

    # df = df[df['outcome'] == "true"]
    # item = np.asarray(df['selected_item'])
    text = df['contents']
    # clickColH,clickColS,clickColL
    itemH,itemS,itemL,itemText = [], [], [], []
    for counter,outcome in enumerate(df['outcome']):
        if outcome and df['condition'][counter] == 'far' and df['role'][counter] == 'speaker':
            itemH.append(df['clickColH'][counter])
            itemS.append(df['clickColS'][counter])
            itemL.append(df['clickColL'][counter])
            itemText.append(text[counter])
    colorsToText = {hsl2rgb([itemH[i], itemS[i]/100, itemL[i]/100]):text[i] for i in range(len(itemH))}
    textToColor = {text[i]:hsl2rgb([itemH[i], itemS[i]/100, itemL[i]/100]) for i in range(len(itemH))}

    print(" ")
    print(colorsToText)
    print(textToColor)

    return colorsToText, textToColor

#3calling wont work
load_item_id_to_utterance_map()

