
import numpy as np

print('starting')
# loads in data from the saved numpy files in dataInput
loadedinputs = np.load('inputs.npy')
loadedtargets = np.load('targets.npy')

# file to save information for when we convert back
filename = open('dataForConversion.txt', 'w')

# TODO: Saved max and min to reset data after NN runs
loadedtargets = loadedtargets.reshape(3072, 32, 32)
# normalization using mean and stddev
for i in range(len(loadedinputs)):
    loadedinputs[i] = (loadedinputs[i] - np.min(loadedinputs[i])) / (np.max(loadedinputs[i]) - np.min(loadedinputs[i]))
for i in range(len(loadedtargets)):
    loadedtargets[i] = (loadedtargets[i] - np.min(loadedtargets[i])) / (np.max(loadedtargets[i]) - np.min(loadedtargets[i]))
print('\n Mean for Targets: ', str(np.mean(loadedtargets)),
      ' Standard Deviation for Targets: ', str(np.std(loadedtargets)), file=filename)

filename.close()

np.save('normedInputs', loadedinputs)
np.save('normedTargets', loadedtargets)
print('done')
