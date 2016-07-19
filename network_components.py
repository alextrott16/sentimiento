import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class Dropout_layer:
    """A layer to perform dropout. Super simple."""

    def __init__(self, n_units, dropout=0.5):
        """
        Initializes the dropout layer
        Parameters
        ----------
        n_units: Number of units (vector size) to apply dropout to
        dropout: Dropout fraction (0 = no dropout)
        """
        err_message = 'Dropout fraction must in the domain [0, 1)'
        assert 0.0 <= dropout < 1.0, err_message
        self.__dropout = dropout

        self.__n_units = n_units

        # Initialize a random stream
        self.rng = RandomStreams().uniform((n_units, 1))

    def mask(self):
        """Generates a theano-friendly mask that applies dropout"""
        # Binarize a random draw to create a Bernoulli distribution
        return T.ceil(self.rng - self.__dropout)

    @property
    def get_dropout(self):
        """Return the dropout parameter"""
        return self.__dropout


class LSTM_layer:
    """A layer of an LSTM network"""

    def __init__(self, num_inputs=None, num_hidden=None, dropout=0.5, c_clip=1000000.):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.__dropout = dropout
        self.curr_mask = None
        self.null_mask = None
        self.null_output_mask = None
        self.initialized = False
        c_clip_val = np.maximum(0., np.abs( c_clip*np.ones((1, 1)).astype(theano.config.floatX)))
        self.c_clip = theano.shared(c_clip_val, broadcastable=(True, True))
        
        # EXPERIMENTAL
        self.g_clip = 10e50

        # Initialize my dropout layer
        self.dropout_layer = Dropout_layer(num_hidden, dropout=dropout)

        # Initialize!
        self.initialize_weights()
        
    def set_c_clip(self, c_clip):
        new_val = np.maximum(0, np.abs(c_clip*np.ones((1, 1)).astype(theano.config.floatX)))
        self.c_clip.set_value(new_val)

    def set_weights(self, W_i, b_i, W_f, b_f, W_o, b_o):
        """
        :param W_i: LSTM input gate weights
        :param b_i: LSTM input gate bias
        :param W_f: LSTM forget gate weights
        :param b_f: LSTM forget gate bias
        :param W_o: LSTM output gate weights
        :param b_o: LSTM output gate bias
        :return: None
        """

        self.W_i = W_i
        self.b_i = b_i

        self.W_f = W_f
        self.b_f = b_f

        self.W_o = W_o
        self.b_o = b_o

        self.W_y = W_y
        self.b_y = b_y

    def initialize_weights(self, num_inputs=None, num_hidden=None,
                           b_i_offset=0., b_f_offset=0., b_c_offset=0., b_o_offset=0.):
        """
        :param num_inputs: number of input units
        :param num_hidden: number of hidden units
        :return: None
        """
        # Handle arguments. I dislike how Matlabian this is. But apparently default arguments are evaluated when a
        # function is declared, so you can't set the default argument to be a class attribute. Bummer.
        if num_inputs is None:
            num_inputs = self.num_inputs
        else:
            self.num_inputs = num_inputs

        if num_hidden is None:
            num_hidden = self.num_hidden
        else:
            self.num_hidden = num_hidden

        # LSTM layers have, for every hidden "unit" a unit and a corresponding memory cell
        # Memory cells include input, forget, and output gates as well as a value
        # Fuck that's a lot of stuff.
        # (this should help):

        # Initialize attributes for every weight of i
        W_i_size = (num_inputs + num_hidden,  # (inp + prev_hidden)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_i)
            self.reset_b(self.b_i, b_i_offset)
        else:
            self.W_i = self.__init_W__(*W_i_size)
            self.b_i = self.__init_b__(num_hidden, b_i_offset)

        # Initialize attributes for every weight of f
        W_f_size = (num_inputs + num_hidden,  # (inp + prev_hidden)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_f)
            self.reset_b(self.b_f, b_f_offset)
        else:
            self.W_f = self.__init_W__(*W_f_size)
            self.b_f = self.__init_b__(num_hidden, b_f_offset)

        # Initialize attributes for every weight of c
        W_c_size = (num_inputs + num_hidden,  # (inp + prev_hidden)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_c)
            self.reset_b(self.b_c, b_c_offset)
        else:
            self.W_c = self.__init_W__(*W_c_size)
            self.b_c = self.__init_b__(num_hidden, b_c_offset)

        # Initialize attributes for every weight of o
        W_o_size = (num_inputs + num_hidden,  # (inp + prev_hidden)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_o)
            self.reset_b(self.b_o, b_o_offset)
        else:
            self.W_o = self.__init_W__(*W_o_size)
            self.b_o = self.__init_b__(num_hidden, b_o_offset)

        # Initialize mask (ONLY IF YOU HAVEN'T ALREADY!!!)
        if not self.initialized:
            self.initialize_masks()

        self.initialized = True

        # Congrats. Now this is initialized.

    @staticmethod
    def __init_W__(n_in, n_out):
        return theano.shared(fan_in_out_uniform(n_in, n_out))

    @staticmethod
    def __init_b__(n, offset):
        # This is, effectively a vector, but we have to make it n-by-1 to enable broadcasting and batch processing
        return theano.shared(
            (offset*np.ones((n, 1))).astype(theano.config.floatX), broadcastable=(False, True))

    @staticmethod
    def reset_W(w):
        w_shape = w.get_value().shape
        w.set_value(fan_in_out_uniform(w_shape[1], w_shape[0]))

    @staticmethod
    def reset_b(b, offset):
        b_shape = b.get_value().shape
        b.set_value((offset*np.ones(b_shape)).astype(theano.config.floatX))

    def initialize_masks(self):
        self.curr_mask = theano.shared(np.ones(shape=(1, self.num_hidden)).astype(theano.config.floatX),
                                       broadcastable=(True, False))
        self.null_mask = theano.shared(np.ones(shape=(self.num_hidden, 1)).astype(theano.config.floatX),
                                       broadcastable=(False, True))

    def list_params(self):
        # Provide a list of all parameters to train
        return [self.W_i, self.b_i, self.W_f, self.b_f, self.W_c, self.b_c, self.W_o, self.b_o]

    def grad_clipper(self, w):
        return theano.gradient.grad_clip(w, -self.g_clip, self.g_clip)
    
    # Write methods for calculating the value of each of these playas at a given step
    def calc_i(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.grad_clipper(self.W_i), combined_inputs) + self.b_i)

    def calc_f(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.grad_clipper(self.W_f), combined_inputs) + self.b_f)
    
    def calc_u(self, combined_inputs):
        return T.tanh(T.dot(self.grad_clipper(self.W_c), combined_inputs) + self.b_c)

    def calc_c(self, prev_c, curr_f, curr_i, curr_u):
        return curr_f*prev_c + curr_i*curr_u

    def calc_o(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.grad_clipper(self.W_o), combined_inputs) + self.b_o)

    def calc_h(self, curr_o, curr_c):
        return curr_o * T.tanh(curr_c)

    def step(self, inp, prev_c, prev_h, mask):
        # Put this together in a method for updating c, and h
        cat_inp = T.concatenate([inp, prev_h])
        i = self.calc_i(cat_inp)
        f = self.calc_f(cat_inp)
        u = self.calc_u(cat_inp)
        c = self.calc_c(prev_c, f, i, u)
        o = self.calc_o(cat_inp)
        h = self.calc_h(o, c) * mask  # Mask applies dropout

        return c, h, i, f, o

    def process(self, sequences, mask):
        """
        Processes a batch of sequences
        :param sequences: tensor3() Variable
            Treated as size=(longest_sequence, input_dimension, n_examples)
        :mask: a pre-generated dropout mask from self.dropout_layer
        :return C: sequence of memory cell activations
        :return H: sequence of hidden activations
        :return I: sequence of input gate activations
        :return F: sequence of forget gate activations
        :return O: sequence of output gate activations
        """
        # Initialize outputs C, and H so that they support a variable number of examples
        n_ex = sequences.shape[2]
        out_init = [
            theano.tensor.alloc(np.zeros(1).astype(theano.config.floatX), self.num_hidden,  n_ex),
            theano.tensor.alloc(np.zeros(1).astype(theano.config.floatX), self.num_hidden,  n_ex),
            None,
            None,
            None
            ]

        ([C, H, I, F, O], updates) = theano.scan(fn=self.step,
                                                 sequences=sequences,
                                                 outputs_info=out_init,
                                                 non_sequences=[mask])

        return C, H, I, F, O


class Base_LSTM_stack:
    """A stack of LSTMs (Base class)"""

    def __init__(self, inp_dim, layer_spec_list, inp_dropout=0.5):
        """A super hacky attempt at a general init"""
        # (This is gonna get over-written, so it's just here for convention)
        self.gen_init(inp_dim, layer_spec_list, inp_dropout)

    def gen_init(self, inp_dim, layer_spec_list, inp_dropout=0.5, bi_flag=False):
        # This is the init function we actually want to be inherited. Hacky. I know.
        """
        Manages multiple layers of LSTM

        Parameters
        ----------
        inp_dim: dimensionality of the input
        inp_dropout: dropout fraction for the input
        layer_spec_list: a list of tuples; each element of the list specifies a layer by (hidden_size, dropout_fraction)
        bi_flag: boolean, specifies whether building a bi-directional or standard LSTM stack
        """
        # First, initialize an input dropout layer
        self.inp_dropout_layer = Dropout_layer(inp_dim, inp_dropout)

        # Then, initialize the rest of the layers
        self.out_dim = 0
        self.layers = []
        for K, spec in enumerate(layer_spec_list):
            # If the first layer, set the input dimensionality to the dimensionality of the input to the entire
            # stack. Otherwise, set it to the output of the previous layer.
            if K == 0:
                my_inps = inp_dim
            else:
                my_inps = layer_spec_list[K-1][0]
                if bi_flag:
                    my_inps *= 2

            # Initialize it now
            new_layer = LSTM_layer(my_inps, spec[0], dropout=spec[1])
            self.layers = self.layers + [new_layer]

            self.out_dim += spec[0]
            if bi_flag:
                self.out_dim += spec[0]

    def initialize_stack_weights(self, b_i_offset=0., b_f_offset=0., b_c_offset=0., b_o_offset=0.):
        """
        Initializes the weights for each layer in the stack
        :return: None
        """
        for i, layer in enumerate(self.layers):
            if type(b_i_offset) == list:
                i_off = b_i_offset[i]
            else:
                i_off = b_i_offset
            if type(b_f_offset) == list:
                f_off = b_f_offset[i]
            else:
                f_off = b_f_offset
            if type(b_c_offset) == list:
                c_off = b_c_offset[i]
            else:
                c_off = b_c_offset
            if type(b_o_offset) == list:
                o_off = b_o_offset[i]
            else:
                o_off = b_o_offset

            layer.initialize_weights(b_i_offset=i_off, b_f_offset=f_off, b_c_offset=c_off, b_o_offset=o_off)

    def list_params(self):
        # Return all the parameters in this stack.... You sure?
        P = []
        for L in self.layers:
            P = P + L.list_params()

        return P


class LSTM_stack(Base_LSTM_stack):
    """A vanilla (forward) LSTM stack"""

    def __init__(self, inp_dim, layer_spec_list, inp_dropout=0.5):
        """
        Manages multiple layers of LSTM

        Parameters
        ----------
        inp_dim: dimensionality of the input
        inp_dropout: dropout fraction for the input
        layer_spec_list: a list of tuples; each element of the list specifies a layer by (hidden_size, dropout_fraction)
        """
        self.bi_flag = False
        self.gen_init(inp_dim, layer_spec_list, inp_dropout, self.bi_flag)

    def process(self, inp_sequences, seq_lengths, test_flag=False):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**
        
        Parameters
        ------
        inp_sequences: tensor3() Variable
            Treated as size=(longest_sequence, input_dimension, n_examples)
        seq_lengths: ivector() Variable
            seq_lengths[i] = The shape[0] of inp_sequences[:,:,i] before zero-padding
            So, it is treated as size=(n_examples,)
        test_flag: Boolean
            Specifies whether to build the graph with train (default) or test/dev dropout
            
        Returns
        -------
        Outputs at end of a given sequence, concatenated across layers
        
            
        """
        # Go through the whole input and return the concatenated outputs of the stack after it's all said and done
        outs = []
        for K, layer in enumerate(self.layers):
            if K == 0:
                if test_flag:
                    # Use the fixed dropout mask
                    curr_seq = inp_sequences * (1 - self.inp_dropout_layer.get_dropout)
                else:
                    # Use the random stream
                    curr_seq = inp_sequences * self.inp_dropout_layer.mask()[None, :, :]
            else:
                curr_seq = last_H  # (from previous layer, dropout already applied)
            
            # Process the sequence for the next dude
            if test_flag:
                mask = (1 - layer.dropout_layer.get_dropout)
            else:
                mask = layer.dropout_layer.mask()
            _, H, _, _, _ = layer.process(curr_seq, mask)
            last_H = H
            
            # Return, for each example, only the final H -- where "final" refers to the true sequence length
            outs = outs + [last_H[seq_lengths-1, :, T.arange(curr_seq.shape[2])]]

        # Transpose so that we are consistent with things expecting n_dim-by-n_examples
        return T.transpose(T.concatenate(outs, axis=1))


class BiLSTM_stack(Base_LSTM_stack):
    """A Bi-directional LSTM stack"""

    def __init__(self, inp_dim, layer_spec_list, inp_dropout=0.5):
        """
        Manages multiple layers of LSTM

        Parameters
        ----------
        inp_dim: dimensionality of the input
        inp_dropout: dropout fraction for the input
        layer_spec_list: a list of tuples; each element of the list specifies a layer by (hidden_size, dropout_fraction)
        """
        self.bi_flag = True
        self.gen_init(inp_dim, layer_spec_list, inp_dropout, self.bi_flag)

    def process(self, inp_sequences, seq_lengths, test_flag=False):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**

        Parameters
        ------
        inp_sequences: tensor4() Variable
            Treated as size=(longest_sequence, input_dimension, n_examples, 2)
            This a concatenation along the 4th dimension of two tensor3 matrices
            inp_sequences[:, :, :, 0] would be an appropriate input to a vanilla LSTM stack
            inp_sequences[:, :, :, 1] is the input sequence oppositely zero padded and reversed
        seq_lengths: ivector() Variable
            seq_lengths[i] = The shape[0] of inp_sequences[:,:,i] before zero-padding
            So, it is treated as size=(n_examples,)
        test_flag: Boolean
            Specifies whether to build the graph with train (default) or test/dev dropout

        Returns
        -------
        Outputs at end of a given sequence, concatenated across layers


        """
        # Go through the whole input and return the concatenated outputs of the stack after it's all said and done
        outs = []
        for K, layer in enumerate(self.layers):
            if K == 0:
                if test_flag:
                    # Use the fixed dropout mask
                    curr_seq_f = inp_sequences[:, :, :, 0] * (1 - self.inp_dropout_layer.get_dropout)
                    curr_seq_b = inp_sequences[:, :, :, 1] * (1 - self.inp_dropout_layer.get_dropout)
                else:
                    # Use the random stream
                    curr_seq_f = inp_sequences[:, :, :, 0] * self.inp_dropout_layer.mask()[None, :, :]
                    curr_seq_b = inp_sequences[:, :, :, 1] * self.inp_dropout_layer.mask()[None, :, :]
            else:
                curr_seq_f = last_H_f  # (from previous layer, dropout already applied)
                curr_seq_b = last_H_b  # (from previous layer, dropout already applied)

            # Process the sequence for the next dude
            if test_flag:
                mask = (1 - layer.dropout_layer.get_dropout)
            else:
                mask = layer.dropout_layer.mask()
            _, fH, _, _, _ = layer.process(curr_seq_f, mask)
            _, bH, _, _, _ = layer.process(curr_seq_b, mask)
            last_fH = fH
            last_bH = bH

            # Return, for each example, only the final H -- where "final" refers to the true sequence length
            f_out = last_fH[seq_lengths - 1, :, T.arange(curr_seq_f.shape[2])]
            b_out = last_bH[seq_lengths - 1, :, T.arange(curr_seq_b.shape[2])]
            outs = outs + [T.concatenate([f_out, b_out], axis=1)]

            # Set up the input to the next layer (assuming there is one)
            # ... concatenate into the (forward) hidden sequence out of this layer
            last_H_f = T.concatenate([last_fH, last_bH], axis=1)
            # ... create the reverse instance
            mat = T.concatenate(
                [T.zeros([T.max(seq_lengths), last_H_f.shape[1], last_H_f.shape[2]]), last_H_f], axis=0)
            # (trust me, it works)
            last_H_b, _ = theano.scan(
                fn=lambda offset, m, length, max_len: T.transpose(
                    m[max_len + length - 1 - offset, :, T.arange(mat.shape[2])]),
                sequences=[T.arange(T.max(seq_lengths))],
                outputs_info=[None],
                non_sequences=[mat, seq_lengths, T.max(seq_lengths)])

        # Transpose so that we are consistent with things expecting n_dim-by-n_examples
        return T.transpose(T.concatenate(outs, axis=1))


class soft_reader:
    """A softmax layer"""

    def __init__(self, num_inputs, num_outputs):
        # This is a simple layer, described just by a single weight matrix and bias
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.w = theano.shared(fan_in_out_uniform(num_inputs, num_outputs))

        # This is, effectively a vector, but we have to make it n-by-1 to enable broadcasting and batch processing
        self.b = theano.shared(
            np.ones((num_outputs, 1)).astype(theano.config.floatX), broadcastable=(False, True))

    def initialize_weights(self):
        w_shape = self.w.get_value().shape
        self.w.set_value(fan_in_out_uniform(w_shape[1], w_shape[0]))

    def list_params(self):
        # Easy.
        return [self.w, self.b]

    def process(self, inp):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**
        
        Inputs
        ------
        inp_sequences: dmatrix() Variable
            Treated as size=(inp_dimension, num_examples)  <--- BATCH PROCESSING
            
        Outputs
        -------
        Outputs a Theano Variable
            Treated as size=(num_examples, num_outputs) <--- Each row sums to 1 (categorical_crossentropy wants this)
        """
        # Do your soft max kinda thing.
        p = T.transpose(T.dot(self.w, inp) + self.b)
        return T.nnet.softmax(p)


class single_class_sigmoid:

    def __init__(self, num_inputs):
        """
        Performs a dot product and adds a bias, into a sigmoid, producing a scalar output

        Parameters
        ----------
        num_inputs: Dimensionality of the input (a matrix of size num_inputs-x-num_examples)
        """
        self.num_inputs = num_inputs
        self.initialize_weights()

    def initialize_weights(self):
        self.w = theano.shared(fan_in_out_uniform(self.num_inputs, 1))

        self.b = theano.shared(
            (np.zeros((1, 1))).astype(theano.config.floatX), broadcastable=(True, True)
        )

    def list_params(self):
        return [self.w, self.b]

    def process(self, inp):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**

        Inputs
        ------
        inp: dmatrix() Variable
            Treated as size=(inp_dimension, num_examples)  <--- BATCH PROCESSING

        Outputs
        -------
        Outputs a Theano Variable
            Treated as size=(num_examples) <--- Scalar output for each example
        """
        return T.nnet.sigmoid(T.dot(self.w, inp) + self.b).flatten()


def ortho_weight(n_in, n_out):
    W = np.random.randn(n_out, n_in)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def fan_in_out_uniform(n_in, n_out):
    return np.random.uniform(
        low=-4 * np.sqrt(6. / (n_in + n_out)),
        high=4 * np.sqrt(6. / (n_in + n_out)),
        size=(n_out, n_in)).astype(theano.config.floatX)
