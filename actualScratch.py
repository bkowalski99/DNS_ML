
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def look(data1, data2):
    for i in range(1024):
        print(i)
        print(np.max(data1[i]))
        print(np.min(data1[i]))
        print(np.average(data1[i]))
        print(np.max(data2[i]))
        print(np.min(data2[i]))
        print(np.average(data2[i]))


# loads in normalized version of data
loadedinputs = np.load('normedInputs.npy')
loadedtargets = np.load('normedTargets.npy')

loadedinputs = loadedinputs.reshape(3072, 32, 32, 1)
loadedtargets = loadedtargets.reshape(3072, 1024, 1)

all_data = tf.data.Dataset.from_tensors((loadedinputs, loadedtargets))

tfds.Split.TRAIN
