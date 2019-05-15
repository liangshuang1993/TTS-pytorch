import unittest
import torch as T

from utils.generic_utils import save_checkpoint, save_best_model
from layers.tacotron import Prenet, CBHG, Decoder, Encoder

OUT_PATH = '/tmp/test.pth.tar'


class ModelSavingTests(unittest.TestCase):
    def save_checkpoint_test(self):
        # create a dummy model
        model = Prenet(128, out_features=[256, 128])
        model = T.nn.DataParallel(layer)

        # save the model
        save_checkpoint(model, None, 100, OUTPATH, 1, 1)

        # load the model to CPU
        model_dict = torch.load(
            MODEL_PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_dict['model'])

    def save_best_model_test(self):
        # create a dummy model
        model = Prenet(256, out_features=[256, 256])
        model = T.nn.DataParallel(layer)

        # save the model
        best_loss = save_best_model(model, None, 0, 100, OUT_PATH, 10, 1)

        # load the model to CPU
        model_dict = torch.load(
            MODEL_PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(model_dict['model'])
