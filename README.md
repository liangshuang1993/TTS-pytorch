<p align="center"><img src="https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png" data-canonical-src="![TTS banner](https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png =250x250)
" width="320" height="95" /></p>

This project is a part of [Mozilla Common Voice](https://voice.mozilla.org/en). TTS aims a deep learning based Text2Speech engine, low in cost and high in quality. To begin with, you can hear a sample generated voice from [here](https://soundcloud.com/user-565970875/commonvoice-loc-sens-attn).

The model architecture is highly inspired by Tacotron: [A Fully End-to-End Text-To-Speech Synthesis Model](https://arxiv.org/abs/1703.10135). However, it has many important updates that make training faster and computationally very efficient. Feel free to experiment with new ideas and propose changes.

You can find [here](http://www.erogol.com/text-speech-deep-learning-architectures/) a brief note about TTS architectures and their comparisons.

## Requirements and Installation
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python>=3.6
  * pytorch>=0.4.1
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode

Install TTS using ```setup.py```. It will install all of the requirements automatically and make TTS available to all the python environment as an ordinary python module.

```python setup.py develop```

Or you can use ```requirements.txt``` to install the requirements only.

```pip install -r requirements.txt```

### Docker
A barebone `Dockerfile` exists at the root of the project, which should let you quickly setup the environment. By default, it will start the server and let you query it. Make sure to use `nvidia-docker` to use your GPUs. Make sure you follow the instructions in the [`server README`](server/README.md) before you build your image so that the server can find the model within the image.

```
docker build -t mozilla-tts .
nvidia-docker run -it --rm -p 5002:5002 mozilla-tts
```

## Checkpoints and Audio Samples
Check out [here](https://mycroft.ai/blog/available-voices/#the-human-voice-is-the-most-perfect-instrument-of-all-arvo-part) to compare the samples (except the first) below.

| Models        |Dataset | Commit            | Audio Sample  | Details |
| ------------- |:------:|:-----------------:|:--------------|:--------|
| [iter-62410](https://drive.google.com/open?id=1pjJNzENL3ZNps9n7k_ktGbpEl6YPIkcZ)|LJSpeech| [99d56f7](https://github.com/mozilla/TTS/tree/99d56f7e93ccd7567beb0af8fcbd4d24c48e59e9)           | [link](https://soundcloud.com/user-565970875/99d56f7-iter62410 )|First model with plain Tacotron implementation.|
| [iter-170K](https://drive.google.com/open?id=16L6JbPXj6MSlNUxEStNn28GiSzi4fu1j) |LJSpeech| [e00bc66](https://github.com/mozilla/TTS/tree/e00bc66) |[link](https://soundcloud.com/user-565970875/april-13-2018-07-06pm-e00bc66-iter170k)|More stable and longer trained model.|
| [iter-270K](https://drive.google.com/drive/folders/1Q6BKeEkZyxSGsocK2p_mqgzLwlNvbHFJ?usp=sharing)|LJSpeech|[256ed63](https://github.com/mozilla/TTS/tree/256ed63)|[link](https://soundcloud.com/user-565970875/sets/samples-1650226)|Stop-Token prediction is added, to detect end of speech.|
| [iter-120K](https://drive.google.com/open?id=1A5Hr6aSvfGgIiE20mBkpzyn3vvbR2APj) |LJSpeech| [bf7590](https://github.com/mozilla/TTS/tree/bf7590) | [link](https://soundcloud.com/user-565970875/sets/september-26-2018-bf7590) | Better for longer sentences |
|[iter-108K](https://drive.google.com/open?id=1cAjRy6jB_3iwRSzkLhD6LutCTOQV28yV)| TWEB | [2810d57](https://github.com/mozilla/TTS/tree/2810d57) | [link](https://soundcloud.com/user-565970875/tweb-example-108k-iters-2810d57) | https://github.com/mozilla/TTS/issues/22 | 
| Best: [iter-185K](https://drive.google.com/open?id=1GU8WGix98WrR3ayjoiirmmbLUZzwg4n0) | LJSpeech | [db7f3d3](https://github.com/mozilla/TTS/tree/db7f3d3) | [link](https://soundcloud.com/user-565970875/sets/ljspeech-model-185k-iters-commit-db7f3d3) | [link](https://github.com/mozilla/TTS/issues/108) |

## Example Model Outputs
Below you see model state after 16K iterations with batch-size 32.

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio output: [https://soundcloud.com/user-565970875/iter16k-f48c3b](https://soundcloud.com/user-565970875/iter16k-f48c3b)

![example_model_output](images/example_model_output.png?raw=true)

## Runtime
The most time-consuming part is the vocoder algorithm (Griffin-Lim) which runs on CPU. By setting its number of iterations, you might have faster execution with a small loss of quality. Some of the experimental values are below.

Sentence: "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

Audio length is approximately 6 secs.

| Time (secs) | System | # GL iters |
| ---- |:-------|:-----------|
|2.00|GTX1080Ti|30|
|3.01|GTX1080Ti|60|


## Datasets and Data-Loading
TTS provides a generic dataloder easy to use for new datasets. You need to write an adaptor to format and that's all you need.Check ```datasets/preprocess.py``` to see example adaptors. After you wrote an adaptor, you need to set ```dataset``` field in ```config.json```. Do not forget other data related fields. 

You can also use pre-computed features. In this case, compute features with ```extract_features.py``` and set ```dataset``` field as ```tts_cache```. 

Example datasets, we successfully applied TTS, are linked below.

- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- [Nancy](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
- [TWEB](http://https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset)
- [M-AI-Labs](http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)

## Training and Fine-tuning LJ-Speech
[Click Here](https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b) for hands-on **Notebook example**, training LJSpeech.

Split ```metadata.csv``` into train and validation subsets respectively ```metadata_train.csv``` and ```metadata_val.csv```. Note that having a validation split does not work well as oppose to other ML problems since at the validation time model generates spectrogram slices without "Teacher-Forcing" and that leads misalignment between the ground-truth and the prediction. Therefore, validation loss does not really show the model performance. Rather, you might use all data for training and check the model performance by relying on human inspection.

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 1100 metadata_shuf.csv > metadata_val.csv
```

To train a new model, you need to define your own ```config.json``` file (check the example) and call with the command below.

```train.py --config_path config.json```

To fine-tune a model, use ```--restore_path```.

```train.py --config_path config.json --restore_path /path/to/your/model.pth.tar```

For multi-GPU training use ```distribute.py```. It enables process based multi-GPU training where each process uses a single GPU.

```CUDA_VISIBLE_DEVICES="0,1,4" distribute.py --config_path config.json```

Each run creates a new output folder and ```config.json``` is copied under this folder.

In case of any error or intercepted execution, if there is no checkpoint yet under the output folder, the whole folder is going to be removed.

You can also enjoy Tensorboard,  if you point the Tensorboard argument```--logdir``` to the experiment folder.

## Testing
Best way to test your network is to use Notebooks under ```notebooks``` folder.

## Contact/Getting Help
- [Wiki](https://github.com/mozilla/TTS/wiki)

- [Discourse Forums](https://discourse.mozilla.org/c/tts) - If your question is not addressed in the Wiki, the Discourse Forums is the next place to look. They contain conversations on General Topics, Using TTS, and TTS Development.

- [Issues](https://github.com/mozilla/TTS/issues) - Finally, if all else fails, you can open an issue in our repo.

## What is new with TTS
If you train TTS with LJSpeech dataset, you start to hear reasonable results after 12.5K iterations with batch size 32. This is the fastest training with character-based methods up to our knowledge. Out implementation is also quite robust against long sentences.

- Location sensitive attention ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention is a vital part of text2speech models. Therefore, it is important to use an attention mechanism that suits the diagonal nature of the problem where the output strictly aligns with the text monotonically. Location sensitive attention performs better by looking into the previous alignment vectors and learns diagonal attention more easily. Yet, I believe there is a good space for research at this front to find a better solution.
- Attention smoothing with sigmoid ([ref](https://arxiv.org/pdf/1506.07503.pdf)). Attention weights are computed by normalized sigmoid values instead of softmax for sharper values. That enables the model to pick multiple highly scored inputs for alignments while reducing the noise.
- Weight decay ([ref](http://www.fast.ai/2018/07/02/adam-weight-decay/)). After a certain point of the training, you might observe the model over-fitting. That is, the model is able to pronounce words probably better but the quality of the speech quality gets lower and sometimes attention alignment gets disoriented.
- Stop token prediction with an additional module. The original Tacotron model does not propose a stop token to stop the decoding process. Therefore, you need to use heuristic measures to stop the decoder. Here, we prefer to use additional layers at the end to decide when to stop.
- Applying sigmoid to the model outputs. Since the output values are expected to be in the range [0, 1], we apply sigmoid to make things easier to approximate the expected output distribution.
- Phoneme based training is enabled for easier learning and robust pronunciation. It also makes easier to adapt TTS to the most languages without worrying about language specific characters.
- Configurable attention windowing at inference-time for robust alignment. It enforces network to only consider a certain window of encoder steps per iteration.
- Detailed Tensorboard stats for activation, weight and gradient values per layer. It is useful to detect defects and compare networks.
- Constant history window. Instead of using only the last frame of predictions, define a constant history queue. It enables training with gradually decreasing prediction frame (r=5 --> r=1) by only changing the last layer. For instance, you can train the model with r=5 and then fine-tune it with r=1 without any performance loss. It also solves well-known PreNet problem [#50](https://github.com/mozilla/TTS/issues/50). 
- Initialization of hidden decoder states with Embedding layers instead of zero initialization. 

One common question is to ask why we don't use Tacotron2 architecture. According to our ablation experiments, nothing, except Location Sensitive Attention, improves the performance, given the increase in the model size.

Please feel free to offer new changes and pull things off. We are happy to discuss and make things better.

## Problems waiting to be solved.
- Punctuations at the end of a sentence sometimes affect the pronunciation of the last word. Because punctuation sign is attended by the attention module, that forces the network to create a voice signal or at least modify the voice signal being generated for neighboring frames.
- ~~Simpler stop-token prediction. Right now we use RNN to keep the history of the previous frames. However, we never tested, if something simpler would work as well.~~ Yet RNN based model gives more stable predictions.
- Train for better mel-specs. Mel-spectrograms are not good enough to be fed Neural Vocoder. Easy solution to this problem is to train the model with r=1. However, in this case, model struggles to align the attention.
- irregular words: "minute", "focus", "aren't" etc. Even though ~~it might be solved~~ (Use a better dataset like Nancy or train phonemes enabled.)

## Major TODOs
- [x] Implement the model.
- [x] Generate human-like speech on LJSpeech dataset.
- [x] Generate human-like speech on a different dataset (Nancy) (TWEB).
- [x] Train TTS with r=1 successfully.
- [x] Enable process based distributed training. Similar [to] (https://github.com/fastai/imagenet-fast/).
- [ ] Adapting Neural Vocoder. The most active work is [here] (https://github.com/erogol/WaveRNN)
- [ ] Multi-speaker embedding.

## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)

### Precursor implementations
- https://github.com/keithito/tacotron (Dataset and Test processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
