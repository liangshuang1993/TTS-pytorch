import os
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import argparse
import importlib

from utils.text import phonemes, symbols
from utils.generic_utils import load_config, sequence_mask
from utils.audio import AudioProcessor
from datasets.TTSDataset import MyDataset
from distribute import DistributedSampler
from models.tacotron import Tacotron


def setup_loader(c, is_val=False, verbose=False):
    global ap
    num_gpus = torch.cuda.device_count()
    
    if is_val and not c.run_eval:
        loader = None
    else:
        preprocessor = importlib.import_module('datasets.preprocess')
        preprocessor = getattr(preprocessor, c.dataset.lower())
        dataset = MyDataset(
            c.data_path,
            c.meta_file_val if is_val else c.meta_file_train,
            c.r,
            c.text_cleaner,
            preprocessor=preprocessor,
            ap=ap,
            batch_group_size=0 if is_val else c.batch_group_size * c.batch_size,
            min_seq_len=0 if is_val else c.min_seq_len,
            max_seq_len=float("inf") if is_val else c.max_seq_len,
            cached=False if c.dataset != "tts_cache" else True,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader

def get_duration(alignment):
    t, s = alignment.shape # t is phoneme's length, s is spectrogram's length
    d = np.zeros(t)
    max_index = np.argmax(alignment, axis=0)
    print(max_index.shape) # max_index's length should be alignment.shape[0]

    for index in max_index:
        d[index] += 1
    
    return d



if __name__ == '__main__':
    # use ground truth mel to get alignment

    parser = argparse.ArgumentParser(description='synthesis parameters')
    parser.add_argument('--root', help='Please input root path', required=True)
    parser.add_argument('--step', help='Please input step', required=False)
    args = parser.parse_args()

    ROOT_PATH = args.root
    MODEL_PATH_TMP = ROOT_PATH + '/checkpoint_{}.pth.tar'

    if args.step is None:
        MODEL_PATH = ROOT_PATH + 'best_model.pth.tar'
    else:
        MODEL_PATH = MODEL_PATH_TMP.format(args.step)

    print(MODEL_PATH)
    CONFIG_PATH = ROOT_PATH + '/config.json'
    OUT_FOLDER = ROOT_PATH + '/test/'

    c = load_config(CONFIG_PATH)
    ap = AudioProcessor(**c.audio)
    use_cuda = True
    
    data_loader = setup_loader(c, is_val=True)
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)
    
    model = Tacotron(
        num_chars=num_chars,
        embedding_dim=c.embedding_size,
        linear_dim=ap.num_freq,
        mel_dim=ap.num_mels,
        r=c.r,
        memory_size=c.memory_size)
    model.eval()

    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                print(num_iter)
                start_time = time.time()

                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                linear_input = data[2]
                mel_input = data[3]
                mel_lengths = data[4]
                stop_targets = data[5]

                # set stop targets view, we predict a single stop token per r frames prediction
                stop_targets = stop_targets.view(text_input.shape[0],
                                                    stop_targets.size(1) // c.r,
                                                    -1)
                stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda(non_blocking=True)
                    text_lengths = text_lengths.cuda(non_blocking=True)
                    mel_input = mel_input.cuda(non_blocking=True)
                    mel_lengths = mel_lengths.cuda(non_blocking=True)
                    linear_input = linear_input.cuda(non_blocking=True)
                    stop_targets = stop_targets.cuda(non_blocking=True)
                mask = sequence_mask(text_lengths)

                # forward pass
                mel_output, linear_output, alignments, stop_tokens =\
                    model.forward(text_input, mel_input, mask)

                for i, alignment in enumerate(alignments):
                    duration = get_duration(alignment)
                    np.save(os.path.join('durations', 'duration-{}.npy'.format(num_iter * c.batch_size + i)), duration)
