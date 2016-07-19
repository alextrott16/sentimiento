import numpy as np
import time
import theano
import theano.tensor as T
import network_optimizers


class network:
    def __init__(self, stack, out_layer, bi_flag, fine_grained,
                 batch_size=25, eval_dev_every=50, alpha=0.005):
        """
        Initializes a self-contained network out of an LSTM stack and a output layer

        Parameters
        ----------
        stack: An LSTM stack object (from network_components.py)
        out_layer: An output layer (from network_components.py)
        bi_flag: Boolean, is this a bi-directional LSTM
        fine_grained: Boolean, is this trained towards the fine-grained task
        batch_size: {25} The batch size to be used for training
        eval_dev_every: {50} Specifies the frequency (in number of training iterations) to check dev accuracy
        alpha: {0.005} The learning rate
        """

        self.fine_grained = fine_grained
        self.bi_flag = bi_flag

        self.stack = stack
        self.out_layer = out_layer

        self.param_list = stack.list_params() + out_layer.list_params()

        self.batch_size = batch_size
        self.eval_dev_every = eval_dev_every

        self.alpha = alpha

        # Build a training and testing graph
        if bi_flag:
            inp_seqs = T.tensor4()
        else:
            inp_seqs = T.tensor3()

        seq_lens = T.ivector()

        # ... training
        s_out_train = stack.process(inp_seqs, seq_lens, test_flag=False)
        out_train = out_layer.process(s_out_train)

        # ... testing
        s_out_test = stack.process(inp_seqs, seq_lens, test_flag=True)
        out_test = out_layer.process(s_out_test)

        # Finish the cost graphs
        if fine_grained:
            targs = T.ivector()
            cost_train = T.nnet.categorical_crossentropy(out_train, targs).mean()
            cost_test = T.nnet.categorical_crossentropy(out_test, targs).mean()
        else:
            targs = T.dvector()
            cost_train = T.nnet.binary_crossentropy(out_train, targs).mean()
            cost_test = T.nnet.binary_crossentropy(out_test, targs).mean()

        # (for training, include a regularization cost)
        cost_train += .0001 * T.sqrt(T.sum(T.concatenate([p.flatten() for p in self.param_list[::2]]) ** 2))

        # The testing functions are most useful for performance diagnosis
        self.guess = theano.function([inp_seqs, seq_lens], out_test)
        self.cfun = theano.function([inp_seqs, seq_lens, targs], cost_test)

        # Set up a trainer
        start = time.time()
        adam_helpers, adam_train, adam_param_list, adam_hyperparam_list, adam_grads = \
            network_optimizers.adam_loves_theano(
                [inp_seqs, seq_lens, targs], cost_train, self.param_list, alpha=alpha)
        dur = time.time() - start
        print dur

        self.adam_helpers = adam_helpers
        self.adam_train = adam_train
        self.adam_params = adam_param_list
        self.adam_hyperparam_list = adam_hyperparam_list
        self.adam_grads = adam_grads

        # Take a snapshot of the starting parameters in case you which to reset
        self.init_val_tup = self.snapshot()  # TAKE A SELFIE LOL!

        # Initialize training logs
        self.cost_each_step = []
        self.dev_cost_each_step = []
        #
        self.dev_acc = []
        #
        self.best_dev_acc = 0
        self.best_dev_idx = 0
        self.no_improve_count = 0

    def snapshot(self):
        """Return a 'snapshot' tuple of the current network/training parameters"""
        param_list = self.param_list
        adam_params = self.adam_params
        w_vals = [p.get_value() for p in param_list]
        a_vals = [[p.get_value() for p in p_grp] for p_grp in adam_params]
        return w_vals, a_vals

    def restore(self, val_tup):
        """
        Restore the network to a snapshot
        Parameters
        ----------
        val_tup: the tuple produced by a call to network.snapshot that you wan't restored to
        """
        param_list = self.param_list
        adam_params = self.adam_params

        for p, v in zip(param_list, val_tup[0]):
            p.set_value(v)
        for p_grp, v_grp in zip(adam_params, val_tup[1]):
            for p, v in zip(p_grp, v_grp):
                p.set_value(v)

    def set_alpha(self, alpha):
        """
        Sets the network learning rate
        Parameters
        ----------
        alpha: the learning rate you want to use (must be greater than 0 -- and shouldn't be > .01)
        """
        assert alpha > 0, 'alpha must be greater than 0'

        adam_hp = self.adam_hyperparam_list.get_value()
        adam_hp[0] = alpha
        self.adam_hyperparam_list.set_value(adam_hp)

        self.alpha = alpha

    def reset_training_logs(self):
        """Resets the training logs"""
        self.cost_each_step = []
        self.dev_cost_each_step = []
        #
        self.dev_acc = []
        #
        self.best_dev_acc = 0
        self.best_dev_idx = 0
        self.no_improve_count = 0

    def total_reset(self):
        """Completely resets to the original parameters. Use at your own risk."""
        self.restore(self.init_val_tup)
        self.reset_training_logs()

    def check_accuracy(self, data, n=None):
        """
        Checks accuracy of network according to the supplied data
        Parameters
        ----------
        data: a list of (vector_sequence, score) tuples in the format of the sentiment analysis data set
        n: the number of samples to use (default is all of them)

        Returns
        -------
        Fraction of correct "guesses"
        """
        if n is None:
            n = len(data)
        else:
            n = np.minimum(n, len(data))

        i_seq, s_len, t, _ = self.pull_data(data, n)
        g = self.guess(i_seq, s_len)

        if self.fine_grained:
            correct = np.argmax(g, axis=1) == t
        else:
            correct = np.round(g) == t

        print '{}/{}, {}% correct'.format(np.sum(correct), len(g), np.round(100. * 100. * np.mean(correct)) / 100.)

        return np.mean(correct)

    def pull_data(self, data, batch_size=None):
        """
        Pull some data for training or evaluating accuracy
        Parameters
        ----------
        data: a list of (vector_sequence, score) tuples in the format of the sentiment analysis data set
        batch_size: the number of data examples to pull (default is network.batch_size)

        Returns
        -------
        (exact details depend on the bi-directionality and classification task)
        inp_seqs: zero-padded batch of input sequences that can be used for training
        seq_lens: the actual length of each sequence in inp_seqs (before zero-padding)
        targets: the training targets
        used_idx: the data indices that were sampled
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Extract the vector sequences and their scores
        seqs, scores = zip(*data)

        # Pick examples at random
        n = len(seqs)
        idx = list(np.random.permutation(np.arange(n)))

        # Initialize the outputs
        inp_seqs = [None] * batch_size
        seq_lens = np.array([0] * batch_size, dtype='int32')
        used_idx = np.array([0] * batch_size, dtype='int32')
        if self.fine_grained:  # Fine_grained (5-bin) classification
            targets = np.array([0] * batch_size, dtype='int32')

            # Use all the data
            def include(score):
                return True
        else:  # Binary classification
            targets = np.zeros(batch_size)

            # Exclude neutral reviews
            def include(score):
                return score <= 0.4 or score > 0.6

        # Collect as much data as you want, until you can't
        pos = 0
        while pos < batch_size and len(idx) > 0:
            # Grab a candidate index
            i = idx.pop()
            # Use it if it can be included
            if include(scores[i]):
                inp_seqs[pos] = seqs[i]
                seq_lens[pos] = seqs[i].shape[0]
                used_idx[pos] = i
                # Bin or binarize
                if self.fine_grained:
                    targets[pos] = np.minimum(4, np.floor(5 * scores[i])).astype('int32')
                else:
                    targets[pos] = np.round(scores[i])
                pos += 1

        # Prune unfilled spots
        if pos < batch_size:
            inp_seqs = inp_seqs[:pos]
            seq_lens = seq_lens[:pos]
            targets = targets[:pos]
            used_idx = used_idx[:pos]

        # Zero-pad the input for batch-processing
        max_length = max(seq_lens)
        pad_seqs = [np.pad(s, ((0, max_length - l), (0, 0)), mode='constant')[:, :, None]
                    for s, l in zip(inp_seqs, seq_lens)]
        # extra steps for bi-directional variant
        if self.bi_flag:
            pad_seqs_B = [np.pad(s, ((max_length - l, 0), (0, 0)), mode='constant')[::-1, :, None]
                          for s, l in zip(inp_seqs, seq_lens)]
            for i in range(len(pad_seqs)):
                pad_seqs[i] = np.concatenate([pad_seqs[i][:, :, :, None], pad_seqs_B[i][:, :, :, None]], axis=3)

        inp_seqs = np.concatenate(pad_seqs, axis=2)

        return inp_seqs, seq_lens, targets, used_idx
