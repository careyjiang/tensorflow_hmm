"""
An example using both tensorflow and numpy implementations of viterbi of ctc
"""
from __future__ import print_function

__author__ = 'carey'

import tensorflow as tf
import numpy as np
import pdb

from hmm_tf_batch import HMMTensorflow


def dptable(V, pathScores, states):
    print(" ".join(("%10d" % i) for i in range(V.shape[0])))
    for i, y in enumerate(pathScores.T):
        print("%.7s: " % states[i])
        print(" ".join("%.7s" % ("%f" % yy) for yy in y))


def main():
    p0 = np.array([0.5, 0.5, 0, 0, 0, 0, 0], dtype=np.float32) #7 states
    
    #_ C _ A _ T _  #[7*10]
    emi = np.array([[0.95, 0.07, 0.35, 0.97, 0.61, 0.48, 0.3,  0.95, 0.03, 0.96],#blank
                    [0.03, 0.9 , 0.5 , 0.01, 0.  , 0.  , 0. ,  0.  , 0.  , 0.  ],#C
                    [0.95, 0.07, 0.35, 0.97, 0.61, 0.48, 0.3,  0.95, 0.03, 0.96],
                    [0.  , 0.  , 0.  , 0.  , 0.23, 0.29, 0.2,  0.  , 0.  , 0.  ],#A
                    [0.95, 0.07, 0.35, 0.97, 0.61, 0.48, 0.3,  0.95, 0.03, 0.96],
                    [0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0. ,  0.02, 0.95, 0.02],#T
                    [0.95, 0.07, 0.35, 0.97, 0.61, 0.48, 0.3,  0.95, 0.03, 0.96],
                    ], dtype=np.float32)
    emi = emi.T #shape [nB, nT, 2nL]
    trans = np.array([[0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], #7*7
                    [0.0, 0.3, 0.3, 0.3, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    states = {0: '_',1: 'C',2: '_',3: 'A',4: '_',5: 'T', 6: '_'}

    #obs_seq = np.array([2, 4, 6]) #path must is CAT
    
    print()
    print("TensorFlow Example: ")

    tf_model = HMMTensorflow(tf.convert_to_tensor(trans), tf.convert_to_tensor(p0))

    y = tf.convert_to_tensor(emi)
    y = tf.expand_dims(y, axis=0)
    y = tf.tile(y, [2,1,1])   #y shape[nB, nT, nC]

    #y_r = tf.gather(y, [0,3,2,1,4,5,6], axis=-1)
    #y = tf.concat([y, y_r], axis=0)
    
    #pdb.set_trace()
    tf_s_graph, tf_scores_graph = tf_model.viterbi_decode_batched(y)
    tf_s = tf.Session().run(tf_s_graph)
    print('tf_s', tf_s)
    print("Most likely States0: ", [states[s] for s in tf_s[0]])
    print("Most likely States1: ", [states[s] for s in tf_s[1]])

    tf_scores = tf.Session().run(tf_scores_graph)
    pathScores = np.array(np.exp(tf_scores[1]))
    dptable(pathScores, pathScores, states)

    print()
    '''
    print("numpy Example: ")
    #pdb.set_trace()
    np_model = HMMNumpy(trans, p0)
    print("init HMMNumpy")
    y = emi
    np_states, np_scores = np_model.viterbi_decode(y)
    print("Most likely States: ", [states[s] for s in np_states])
    pathScores = np.array(np.exp(np_scores))
    dptable(pathScores, pathScores, states)
    '''
if __name__ == "__main__":
    main()
