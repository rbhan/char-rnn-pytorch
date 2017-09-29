import codecs
import os
import numpy as np
import torch.utils.data


class CharRNNDataset(torch.utils.data.Dataset):
    def __init__(self, text, config):
        self.char_ls = sorted(list(set(text)))
        self.vocab_size = len(self.char_ls)
        self.char_to_int_map = dict(zip(self.char_ls, range(len(self.char_ls))))
        self.int_to_char_map = dict(zip(range(len(self.char_ls)), self.char_ls))

        full_data = np.array(list(map(self.char_to_int_map.get, text)))

        # Truncate slightly so we have a num_chars = num_lines * seq_length
        self.num_lines = int(len(full_data) / config.seq_length)
        full_data = full_data[
            :self.num_lines * config.seq_length
        ]

        full_x_data = full_data.copy()
        full_y_data = full_data.copy()
        full_y_data[:-1] = full_data[1:]
        full_y_data[-1] = full_data[0]

        self.x_lines = full_x_data.reshape(self.num_lines, config.seq_length)
        self.y_lines = full_y_data.reshape(self.num_lines, config.seq_length)

    def __getitem__(self, index):
        x = torch.LongTensor(self.x_lines[index])
        y = torch.LongTensor(self.y_lines[index])
        return x, y

    def __len__(self):
        return self.num_lines


def load_data(config):
    """ Load data and get Dataset and Dataloader """
    input_path = os.path.join(config.data_dir, "input.txt")
    with codecs.open(input_path, "r", encoding="utf-8") as f:
        data = f.read()

    dataset = CharRNNDataset(text=data, config=config)
    config.vocab_size = dataset.vocab_size
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=1, pin_memory=config.cuda,
    )
    return dataset, dataloader
