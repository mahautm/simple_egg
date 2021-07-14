# This is from the basic_game reconstruction setup, with notified minor modifications
# Change in data loader : added sum of element
# Changes in the loss function : modified expected results to be twice less (one result for every two inputs)
from sys import path
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core

from egg.zoo.basic_games.architectures import RecoReceiver, Sender

from egg.zoo.basic_games.play import get_params
from data_reader import AttValSumDataset
from egg.core.callbacks import InteractionSaver, CheckpointSaver
from egg.core.language_analysis import MessageEntropy


def model_setup(opts):
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    if opts.game_type == "sum":

        def loss(
            sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
        ):
            n_attributes = opts.n_attributes
            n_values = opts.n_values
            batch_size = sender_input.size(0)
            receiver_output = receiver_output.view(
                batch_size, n_values * n_attributes
            )  # <-- here be changes, a single value is expected per trial
            receiver_guesses = receiver_output.argmax(dim=1)
            correct_samples = (
                (receiver_guesses == labels.view(-1))
                .view(
                    batch_size
                )  # <-- here be changes, a single value is expected per trial
                .detach()
            )
            acc = (
                torch.sum(correct_samples, dim=-1) / len(correct_samples)
            ).float()  # <-- here be changes, accuracy here is the percentage of sucesses
            labels = labels.view(
                batch_size
            )  # <-- here be changes, a single value is expected per trial
            loss = F.cross_entropy(receiver_output, labels, reduction="none")
            loss = loss.view(batch_size, -1).mean(dim=1)
            return loss, {"acc": acc}

        train_loader = DataLoader(
            AttValSumDataset(
                path=opts.train_data,
                n_attributes=opts.n_attributes,
                n_values=opts.n_values,
            ),
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=1,
        )
        test_loader = DataLoader(
            AttValSumDataset(
                path=opts.validation_data,
                n_attributes=opts.n_attributes,
                n_values=opts.n_values,
            ),
            batch_size=opts.validation_batch_size,
            shuffle=False,
            num_workers=1,
        )

        n_features = opts.n_attributes * opts.n_values
        receiver = RecoReceiver(n_features=n_features, n_hidden=opts.receiver_hidden)
    else:
        warnings.warn(f"{opts.game_type} is not an implemented game type")
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)
    if opts.mode.lower() == "gs":
        # in the following lines, we embed the Sender and Receiver architectures into standard EGG wrappers that are appropriate for Gumbel-Softmax optimization
        # the Sender wrapper takes the hidden layer produced by the core agent architecture we defined above when processing input, and uses it to initialize
        # the RNN that generates the message
        sender = core.RnnSenderGS(
            sender,
            vocab_size=opts.vocab_size,
            embed_dim=opts.sender_embedding,
            hidden_size=opts.sender_hidden,
            cell=opts.sender_cell,
            max_len=opts.max_len,
            temperature=opts.temperature,
        )
        # the Receiver wrapper takes the symbol produced by the Sender at each step (more precisely, in Gumbel-Softmax mode, a function of the overall probability
        # of non-eos symbols upt to the step is used), maps it to a hidden layer through a RNN, and feeds this hidden layer to the
        # core Receiver architecture we defined above (possibly with other Receiver input, as determined by the core architecture) to generate the output
        receiver = core.RnnReceiverGS(
            receiver,
            vocab_size=opts.vocab_size,
            embed_dim=opts.receiver_embedding,
            hidden_size=opts.receiver_hidden,
            cell=opts.receiver_cell,
        )
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        # callback functions can be passed to the trainer object (see below) to operate at certain steps of training and validation
        # for example, the TemperatureUpdater (defined in callbacks.py in the core directory) will update the Gumbel-Softmax temperature hyperparameter
        # after each epoch
        path = f"/gpfsscratch/rech/imi/ude64um/simple_egg_exp/vo{opts.vocab_size}_ma{opts.max_len}"
        # added all callbacks
        callbacks = [
            core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1),
            InteractionSaver(checkpoint_dir=path),
            CheckpointSaver(checkpoint_path=path, checkpoint_freq=500),
            MessageEntropy(is_gumbel=True),
        ]

    optimizer = core.build_optimizer(game.parameters())

    if opts.print_validation_events == True:
        # we add a callback that will print loss and accuracy after each training and validation pass (see ConsoleLogger in callbacks.py in core directory)
        # if requested by the user, we will also print a detailed log of the validation pass after full training: look at PrintValidationEvents in
        # language_analysis.py (core directory)
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )
    return trainer, game


def main(params):
    opts = get_params(params)
    print(params)
    trainer, _ = model_setup(opts)
    # and finally we train!
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
