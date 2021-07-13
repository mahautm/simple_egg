# Change in data loader : added sum of element
# Changes in the loss function : modified expected results to be twice less (one result for every two inputs)
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core

# from egg.core import Callback, Interaction, PrintValidationEvents
from egg.zoo.basic_games.architectures import RecoReceiver, Sender

# from egg.zoo.basic_games.data_readers import AttValDiscriDataset, AttValRecoDataset
from egg.zoo.basic_games.play import get_params
from data_reader import AttValSumDataset
from egg.core.callbacks import InteractionSaver
from egg.core.language_analysis import MessageEntropy, TopographicSimilarity, Disent


def main(params):
    opts = get_params(params)
    if opts.validation_batch_size == 0:
        opts.validation_batch_size = opts.batch_size
    print(opts, flush=True)

    # the following if statement controls aspects specific to the two game tasks: loss, input data and architecture of the Receiver
    # (the Sender is identical in both cases, mapping a single input attribute-value vector to a variable-length message)
    if opts.game_type == "sum":

        def loss(
            sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
        ):
            # in the case of the recognition game, for each attribute we compute a different cross-entropy score
            # based on comparing the probability distribution produced by the Receiver over the values of each attribute
            # with the corresponding ground truth, and then averaging across attributes
            # accuracy is instead computed by considering as a hit only cases where, for each attribute, the Receiver
            # assigned the largest probability to the correct value
            # most of this function consists of the usual pytorch madness needed to reshape tensors in order to perform these computations
            n_attributes = opts.n_attributes
            n_values = opts.n_values
            batch_size = sender_input.size(0)
            receiver_output = receiver_output.view(
                batch_size, n_values * n_attributes
            )  # <-- here be changes
            receiver_guesses = receiver_output.argmax(dim=1)
            correct_samples = (
                (receiver_guesses == labels.view(-1))
                .view(batch_size)  # <-- here be changes
                .detach()
            )
            acc = (torch.sum(correct_samples, dim=-1) / len(correct_samples)).float()
            labels = labels.view(batch_size)  # <-- here be changes
            loss = F.cross_entropy(receiver_output, labels, reduction="none")
            loss = loss.view(batch_size, -1).mean(dim=1)
            return loss, {"acc": acc}

        # again, see data_readers.py in this directory for the AttValRecoDataset data reading class
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
        # the number of features for the Receiver (input) and the Sender (output) is given by n_attributes*n_values because
        # they are fed/produce 1-hot representations of the input vectors
        n_features = opts.n_attributes * opts.n_values
        # we define here the core of the receiver for the discriminative game, see the architectures.py file for details
        # this will be embedded in a wrapper below to define the full architecture
        receiver = RecoReceiver(n_features=n_features, n_hidden=opts.receiver_hidden)
    else:
        warnings.warn(f"{opts.game_type} is not an implemented game type")
    # we are now outside the block that defined game-type-specific aspects of the games: note that the core Sender architecture
    # (see architectures.py for details) is shared by the two games (it maps an input vector to a hidden layer that will be use to initialize
    # the message-producing RNN): this will also be embedded in a wrapper below to define the full architecture
    sender = Sender(n_hidden=opts.sender_hidden, n_features=n_features)
    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object
    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
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
        callbacks = [
            core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1),
            InteractionSaver(
                checkpoint_dir="/gpfsscratch/rech/imi/ude64um/simple_egg_exp/"
            ),
            MessageEntropy(is_gumbel=True),
            TopographicSimilarity(),
            # Disent(vocab_size=opts.vocab_size, is_gumbel=True),
        ]

    # we are almost ready to train: we define here an optimizer calling standard pytorch functionality
    optimizer = core.build_optimizer(game.parameters())
    # in the following statement, we finally instantiate the trainer object with all the components we defined (the game, the optimizer, the data
    # and the callbacks)
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

    # and finally we train!
    trainer.train(n_epochs=opts.n_epochs)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
