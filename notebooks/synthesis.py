import io
import librosa
import torch
import numpy as np
from utils.text import text_to_sequence
from utils.data import prepare_data
from matplotlib import pylab as plt

hop_length = 250


def create_speech(m, s, CONFIG, use_cuda, ap):
    text_cleaner = [CONFIG.text_cleaner]
    texts = [np.asarray(text_to_sequence(text, text_cleaner), dtype=np.int32) for text in s]
    texts = prepare_data(texts).astype(np.int32)
    
    texts = torch.LongTensor(texts)
    if use_cuda:
        texts = texts.cuda()
    mel_out, linear_outs, alignments, stop_tokens = m.forward(texts.long())
    linear_outs = [linear_out.data.cpu().numpy() for linear_out in linear_outs]
    alignments = [alignment_.cpu().data.numpy() for alignment_ in alignments]
    specs = [ap._denormalize(linear_out) for linear_out in linear_outs]
    wavs = [ap.inv_spectrogram(linear_out.T) for linear_out in linear_outs]
    # wav = wav[:ap.find_endpoint(wav)]
    out = io.BytesIO()
    # ap.save_wav(wav, out)
    return wavs, alignments, specs, stop_tokens


def visualize(alignment, spectrogram, stop_tokens, CONFIG):
    label_fontsize = 16
    plt.figure(figsize=(16, 24))

    plt.subplot(3, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    plt.colorbar()

    stop_tokens = stop_tokens.squeeze().detach().to('cpu').numpy()
    plt.subplot(3, 1, 2)
    plt.plot(range(len(stop_tokens)), list(stop_tokens))

    plt.subplot(3, 1, 3)
    librosa.display.specshow(
        spectrogram.T,
        sr=CONFIG.sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear")
    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)