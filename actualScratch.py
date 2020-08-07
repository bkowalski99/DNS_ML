
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


def heatmapcomparison(data1, data2, numberofplots):
    for z in range(numberofplots):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Graph "+str(z))
        ax1.imshow(data1[z])
        ax1.set_title('input smooth field')
        ax2.imshow(data2[z])
        ax2.set_title('target variance field')
        plt.show()


# loads in normalized version of data
originputs = np.load('inputs.npy')
origtargets = np.load('targets.npy')
loadedinputs = np.load('normedInputs.npy')
loadedtargets = np.load('normedTargets.npy')

originputs = originputs.reshape(3072, 32, 32)
origtargets = origtargets.reshape(3072, 32, 32)
loadedinputs = loadedinputs.reshape(3072, 32, 32)
loadedtargets = loadedtargets.reshape(3072, 32, 32)

heatmapcomparison(originputs, origtargets, 32)
