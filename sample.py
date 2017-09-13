import numpy as np
import torch
from torch.autograd import Variable

from utils import new_hidden, get_sampling_function, to_text, maybe_cuda_var


def sample_model(config, dataset, model,
                 prime_text="The ", length=1000, text=True):
    sampling_func = get_sampling_function(config)

    # Convert prime_text to array
    prime = np.array(
        list(map(dataset.char_to_int_map.get, prime_text))
    ).reshape(1, -1)

    # Initialize state for generation
    hidden_gen = new_hidden(model, batch_size=config.batch_size)
    history = prime

    # Sample character by character
    for i in range(length):
        # Only use the last runner.seq_length for prediction input
        pred_input = history[:, -config.seq_length:]
        pred_input_var = maybe_cuda_var(
            torch.LongTensor(pred_input), cuda=config.cuda)
        output_gen, hidden_gen = model(
            pred_input_var,
            hidden_gen,
        )

        # Get new character-index
        new_ci = sampling_func(np.exp(output_gen[0, -1].data.cpu().numpy()))
        # Add character-index to history
        history = np.hstack([history, [[new_ci]]])

    if text:
        return to_text(history[0], dataset=dataset)
    else:
        return history[0]
