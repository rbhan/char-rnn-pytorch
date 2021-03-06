from __future__ import print_function
import config as char_rnn_config
import train as char_rnn_train
import sample as char_rnn_sample

config = char_rnn_config.Config(
    cuda=False,
    data_dir="data/tinyshakespeare",
    save_dir=None,
    rnn_size=128,
    rnn_model="LSTM",
    num_layers=2,
    batch_size=64,
    seq_length=50,
    num_epochs=50,
    save_every=False,
    grad_clip=5.,
    learning_rate=0.002,
    decay_rate=0.97,
    keep_prob=1.0,
    sampling_mode="weighted",
)

dataset, model, losses = char_rnn_train.train_model(config)

print(char_rnn_sample.sample_model(
    config=config,
    dataset=dataset,
    model=model,
))
