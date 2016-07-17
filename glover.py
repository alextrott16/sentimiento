import re
import numpy as np
import cPickle
import time
import sys
from IPython.display import clear_output


class glove:
    
    
    def __init__(self, glove_txt_file='./data/glove_files/glove_840B_300d.txt'):
        """
        Initialize a lookup scheme for the word vectors described in glove_txt_file
        
        Input: glove_txt_file (filepath to the relevant glove information)
        Output: None
        """
        print 'Preparing information from {} ...'.format(glove_txt_file)
        sys.stdout.flush()
        
        # Import each word and vector as text
        with open(glove_txt_file, 'r') as f:
            GV = [line.strip() for line in f]


        # Determine the dimensionality
        gv = GV[0]
        m = re.match('\S+', gv)
        d = len( re.split('\s', gv[m.end()+1:]) )

        # Initialize your storage dudes
        N = len(GV)
        K = [None]*N
        self.w_mat = np.zeros(shape=(N, d))
        
        # Initialize the unknown vector
        self.n_dims = d
        self.__unknown = np.zeros(d)


        # Step through every word
        start = time.time()
        # (Display the progress every this many)
        disp_every = 5000
        for i, gv in enumerate(GV):
            # Find the word
            m = re.match('\S+',gv)

            # Store the word at this index
            K[i] = m.group()
            # Store the vector at this index
            values = re.split('\s', gv[m.end()+1:])
            self.w_mat[i,] = [np.float(v) for v in values]


            # Display
            if ((i+1) % disp_every) == 0:
                dur = time.time()-start
                pcnt_comp = 100.*i/float(N)
                rate = pcnt_comp/dur
                time_left = (100. - pcnt_comp)/rate

                pc = np.round(10.*pcnt_comp)/10.
                tl = np.round(time_left)
                clear_output()
                print '{}% complete; about {} seconds left.'.format(pc, tl)
                sys.stdout.flush()


        # One last display
        dur = time.time()-start
        clear_output()
        print '100% complete; {} seconds.'.format(np.round(100.*dur)/100.)
        print 'Done!'
        sys.stdout.flush()


        # Final touch
        self.lookup = dict( zip(K, range(len(K))) )
        
        
    def get_unknown(self):
        """
        returns current vector for unknown words
        """
        return self.__unknown
    
    
    def set_unknown(self, unknown):
        """
        used to set the unknown vector
        """
        err_message =  "Argument 'unknown' must be a {}-dimensional numpy vector".format(n_dims)
        assert type(unknown) is np.ndarray,     err_message
        assert unknown.shape == (self.n_dims,), err_message
        
        self.__unknown = unknown
        
        
    def vec(self, string):
        """
        returns the vector embedding of the provided word, or the unknown vector if word is not in dictionary
        """
        if self.lookup.has_key(string):
            return self.w_mat[self.lookup[string]]
        else:
            return self.__unknown