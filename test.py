import os
import sys
import io
import torch 
import time
import numpy as np
from collections import OrderedDict
import argparse
import math

import librosa

from models.tacotron import Tacotron 
from utils.audio import AudioProcessor
from utils.text.symbols import phonemes, symbols

parser = argparse.ArgumentParser(description='synthesis parameters')
parser.add_argument('--root', helper='Please input root path', required=True)
parser.add_argument('--step', helper='Please input step', required=False)
parser.add_argument('--text', helper='Please input text file', default='pinyin.txt')
args = parser.parse_args()


def text2audio(texts, model, CONFIG, use_cuda, ap):
    wavs = []
    wavs, alignments, spectrograms, stop_tokens = create_speech(mode, texts, CONFIG, use_cuda, ap)
    return wavs


ROOT_PATH = args.root
MODEL_PATH_TMP = ROOT_PATH + '/checkpoint_{}.pth.tar'

if args.step is None:
    MODEL_PATH = ROOT_PATH + 'best_model.pth.tar'
else:
    MODEL_PATH = MODEL_PATH_TMP.format(args.step)

print(MODEL_PATH)
CONFIG_PATH = ROOT_PATH + '/config.json'
OUT_FOLDER = ROOT_PATH + '/test/'
if not os.path.exists(OUT_FOLDER):
    os.makedirs(OUT_FOLDER)

CONFIG = load_config(CONFIG_PATH)
use_cuda = True

ap = AudioProcessor(**CONFIG.audio)
num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
model = Tacotron(num_chars, CONFIG.embedding_size, ap.num_freq, ap.num_mels, CONFIG.r)


texts = []

with open(args.text) as f:
    for line in f:
        texts.append(line.strip())

if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
model.decoder.max_decoder_steps = 800
batch_size = 32

for n in range(math.ceil(len(texts) / batch_size)):
    batch_texts = texts[n: max(n + batch_size, len(texts))]
    wavs = text2audio(texts, model, CONFIG, use_cuda, ap)
    for i, wav in enumerate(wavs):
        ap.save_wav(wav, os.path.join(OUT_FOLDER, 'CommonVoice_{}_{}_{}.wav'.format(args.step, n, i)))
