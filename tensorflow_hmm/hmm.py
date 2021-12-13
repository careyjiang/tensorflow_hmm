import tensorflow as tf
import numpy as np

class HMM(object):
    """
    A class for Hidden Markov Models.
    use tf tensor as data struct instead of numpy
    The model attributes are:
    - K :: the number of states
    - P :: the K by K transition matrix (from state i to state j,
        (i, j) in [1..K])
    - p0 :: the initial distribution (defaults to starting in state 0)
    """

    def __init__(self, P, p0=None, length=None):
        self.K = tf.shape(P)[0]
        self.length = length

        #normal
        P = P / tf.reduce_sum(P,1)

        #transition matrix
        self.P = tf.cast(P, tf.float32)
        self.logP = tf.log(self.P)

        #init p0
        self.p0 = p0
        self.logp0 = tf.log(self.p0)

class HMMTensorflow(HMM):

    def _viterbi_partial_forward_batched(self, scores):
        """
        Expects inputs in [B, Kl layout
        """
        # first convert scores into shape [B, K, 1]
        # then concatenate K of them into shape [B, K, K]
        #expanded_scores = tf.concat(
        #    [tf.expand_dims(scores, axis=2)] * self.K, axis=2
        #)
        scores = tf.expand_dims(scores, axis=2)
        expanded_scores = tf.tile(scores,[1, 1, self.K])
        return expanded_scores + self.logP

    def viterbi_decode_batched(self, y):
        """
        Runs viterbi decode on state probabilies y in batch mode

        Arguments
        ---------
        y : tensor : shape (B, T, K) where T is number of timesteps and
            K is the number of states

        Returns
        -------
        (s, pathScores)
        s : list of length T of tensorflow ints : represents the most likely
            state at each time step.
        pathScores : list of length T of tensorflow tensor of length K
            each value at (t, k) is the log likliehood score in state k at
            time t.  sum(pathScores[t, :]) will not necessary == 1
        """
        if len(y.shape) == 2:
            y = tf.expand_dims(y, axis=0)

        if  len(y.shape) != 3:
            raise ValueError((
                'y should be 3d of shape (nB, nT, {}).  Found {}'
            ).format(self.K, y.shape))

        nB = tf.shape(y)[0]
        nT = tf.shape(y)[1]
        nC = tf.shape(y)[2]

        #use tensor
        pathStates = tf.zeros([nB, nT, nC], dtype=tf.int32) #[]
        pathScores = tf.zeros([nB, nT, nC], dtype=tf.float32) #[]

        # initialize
        y_0, _ = tf.split(y, [1, nT-1], 1)
 
        pathScores_0_new = self.logp0 + tf.log(y_0)
        _, pathScores_nT_1 = tf.split(pathScores, [1, nT-1], 1)
        pathScores = tf.concat([pathScores_0_new, pathScores_nT_1], 1)

        def forward_cond(t, nT, y, pathStates, pathScores):
            return t < nT-1
        def forward_body(t, nT, y, pathStates, pathScores):
            _, y_t_1, _ = tf.split(y, [t+1, 1, nT-t-2], 1)
            yy = tf.squeeze(y_t_1, axis=1)
            # propagate forward
            _, pathScores_t, _= tf.split(pathScores, [t, 1, nT-t-1], 1)
            tmpMat = self._viterbi_partial_forward_batched(tf.squeeze(pathScores_t, axis=1))

            # the inferred state
            pathStates_t_new = tf.argmax(tmpMat, axis=1)
            pathStates_t_new = tf.expand_dims(pathStates_t_new, axis=1)
            pathStates_t_new = tf.cast(pathStates_t_new, tf.int32)
            pathStates_t_b1, _, pathStates_t_a1= tf.split(pathStates, [t+1, 1, nT-t-2], 1)
            pathStates = tf.concat([pathStates_t_b1, pathStates_t_new, pathStates_t_a1], 1)
                        
            pathScores_t_new = tf.reduce_max(tmpMat, axis=1) + tf.log(yy)
            pathScores_t_new = tf.expand_dims(pathScores_t_new, axis=1) #[nB, 1, nK]
            pathScores_b1, _, pathScores_t_a1= tf.split(pathScores, [t+1, 1, nT-t-2], 1)
            pathScores = tf.concat([pathScores_b1, pathScores_t_new, pathScores_t_a1], 1) 
            return t+1, nT, y, pathStates, pathScores
        t, nT, y, pathStates, pathScores = tf.while_loop(forward_cond, forward_body,[0, nT, y, pathStates, pathScores])

        _, pathScores_last= tf.split(pathScores, [nT-1, 1], 1)
        s_1 = tf.argmax(tf.squeeze(pathScores_last, axis=1), axis=1)
        s_1 = tf.cast(s_1, tf.int32)
        s_1 = tf.expand_dims(s_1, axis=1)
        s_o = tf.zeros([nB, nT-1], tf.int32)
        s = tf.concat([s_o, s_1], axis=1)
        def backtrack_cond(t, s, s_last, pathStates):
            return t > 0
        def backtrack_body(t, s, s_last, pathStates):
            _, pathStates_t, _= tf.split(pathStates, [t, 1, nT-t-1], 1)
            pathStates_t = tf.squeeze(pathStates_t, axis=1)
            index = tf.range(0, tf.shape(s)[0])
            index = tf.expand_dims(index, axis=1)
            s_last_index = tf.concat([index, s_last], axis = 1)
            s_last_index = tf.expand_dims(s_last_index, axis = 1)
            s_cur = tf.gather_nd(pathStates_t, s_last_index)

            pathScores_b, pathScores_cur, pathScores_a= tf.split(s, [t-1, 1, nT-t], 1)
            s = tf.concat([pathScores_b, s_cur, pathScores_a], axis=1)
            return t-1, s, s_cur, pathStates
        _, s, _, _ =tf.while_loop(backtrack_cond, backtrack_body,[nT - 1, s, s_1, pathStates])
         
        return s, pathScores
