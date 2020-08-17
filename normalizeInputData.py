
""" Author: Ben Kowalski
    Date: 8/17/2020
    Email: bkowalski99@gmail.com
    Function: This script normalizes the 32x32 matrices provided by dataInput.py. Without the inputs.npy and targets.npy
    files this file will fail. The results are stored as normedInputs.npy and normedTargets.npy respectively.
    The maximum and minimum of each plane are saved in the files inputsConversion32x32.txt and
    targetsConversion32x32.txt. To restore these to be non normalized multiply the plane by the difference of the
    maximum and the minimum, then add the minimum to the plane.
"""

import numpy as np

print('starting')
# loads in data from the saved numpy files in dataInput
loadedinputs = np.load('inputs.npy')
loadedtargets = np.load('targets.npy')
# file to save information for when we convert back

loadedinputs = loadedinputs.reshape(3072, 32, 32)
loadedtargets = loadedtargets.reshape(3072, 32, 32)

# normalization uses x - min / max - min
# this makes max possible value 1 and min possible value 0
file = open('inputsConversion32x32.txt', 'w')
print("Normalizing 32 by 32 inputs")
for i in range(len(loadedinputs)):
    print('Run:', str(i), ' max:', np.max(loadedinputs[i]), ' min:', np.min(loadedinputs[i]), file=file)
    loadedinputs[i] = (loadedinputs[i] - np.min(loadedinputs[i])) / (np.max(loadedinputs[i]) - np.min(loadedinputs[i]))

file.close()

file = open('targetsConversion32x32.txt', 'w')

print("Normalizing 32 by 32 targets")
for i in range(len(loadedtargets)):
    print('Run:', str(i), ' max:', np.max(loadedtargets[i]), ' min:', np.min(loadedtargets[i]), file=file)
    loadedtargets[i] = (loadedtargets[i] - np.min(loadedtargets[i])) / (np.max(loadedtargets[i]) - np.min(loadedtargets[i]))

file.close()

np.save('normedInputs', loadedinputs)
np.save('normedTargets', loadedtargets)
print('done')
