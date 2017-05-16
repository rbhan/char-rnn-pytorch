import numpy as np

from utils import get_data_dir
from config import Config
from train import train_model
from sample import sample_model


def test_run():
    config = Config(
        cuda=False,
        data_dir=get_data_dir("gettysburg"),
        save_dir=None,
        rnn_size=2,
        rnn_model="LSTM",
        num_layers=2,
        batch_size=8,
        seq_length=4,
        num_epochs=1,
        save_every=None,
        grad_clip=5.,
        learning_rate=0.002,
        decay_rate=0.97,
        keep_prob=1.0,
        sampling_mode="weighted",
    )
    dataset, model, losses = train_model(config, verbose=False)
    hash_result = np.sum(sample_model(
        config=config,
        dataset=dataset,
        model=model,
        text=False,
    ))
    assert hash_result == 18989, hash_result
