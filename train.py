import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils

from model import CharRNNModel
from data import load_data
from utils import new_hidden, repackage_sample
from sample import sample_model


def train_model(config,
                sample_prime_text="The ", sample_length=50):
    dataset, dataloader = load_data(config)

    model = CharRNNModel(config)
    if config.cuda:
        model = model.cuda()

    # Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )
    losses = []

    # Training Loop
    model.train()
    print "START TRAINING for {} Epochs".format(config.num_epochs)
    for epoch in range(config.num_epochs):
        batch_loss = 0
        for x_batch, y_batch in dataloader:

            # Pre-process inputs
            x_var = repackage_sample(x_batch, config=config)
            y_var = repackage_sample(y_batch, config=config)

            # Run, perform gradient-descent
            model.zero_grad()
            output, hidden = model(
                x=x_var,
                hidden=new_hidden(model, batch_size=config.batch_size),
            )
            loss = criterion(
                output.contiguous().view(-1, config.vocab_size),
                y_var.view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), config.grad_clip)
            optimizer.step()

            # Record loss
            loss_value = loss.data.cpu().numpy()[0]
            batch_loss += loss_value
        losses.append(batch_loss)

        # Print status
        print "Epoch={}/{}: Loss={}".format(
            epoch, config.num_epochs, batch_loss,
        )
        print "Sample: "
        print indent_text(sample_model(
            config=config,
            dataset=dataset,
            model=model,
            prime_text=sample_prime_text,
            length=sample_length,
        ))

        # Decay Learning Rate
        for param_group in optimizer.param_groups:
            param_group['lr'] *= config.decay_rate

    return dataset, model, losses


def indent_text(text, indentation=8):
    return "\n".join([
        " " * indentation + line
        for line in text.splitlines()
    ])
