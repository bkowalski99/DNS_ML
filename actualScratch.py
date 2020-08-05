
import numpy as np
import tensorflow as tf
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

inputset = np.split(loadedinputs, 3)
targetset = np.split(loadedtargets, 3)

# moving data into proper sections
training_inputs = inputset[0]
training_targets = targetset[0]
validation_inputs = inputset[1]
validation_targets = targetset[1]
testing_inputs = inputset[2]
testing_targets = targetset[2]

# reformatting data so ndim=4
training_inputs = training_inputs.reshape(1024, 32, 32, 1)
validation_inputs = validation_inputs.reshape(1024, 32, 32, 1)
testing_inputs = testing_inputs.reshape(1024, 32, 32, 1)
training_targets = training_targets.reshape(1024, 1024, 1)
validation_targets = validation_targets.reshape(1024, 1024, 1)
testing_targets = testing_targets.reshape(1024, 1024, 1)

look(validation_inputs, validation_targets)
