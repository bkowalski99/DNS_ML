import numpy as np

print('starting')
# loads in data from the saved numpy files in dataInput
inputs_8x8 = np.load('8X8_inputs.npy')
targets_8x8 = np.load('8X8_targets.npy')
# file to save information for when we convert back

inputs_8x8 = inputs_8x8.reshape(49152, 8, 8)
targets_8x8 = targets_8x8.reshape(49152, 8, 8)

# normalization uses x - min / max - min
# this makes max possible value 1 and min possible value 0

file = open('inputsConversion8x8.txt', 'w')

print("Normalizing 8 by 8 inputs")
for i in range(len(inputs_8x8)):
    if np.max(inputs_8x8[i]) != np.min(inputs_8x8[i]):
        print('Run:', str(i), ' max:', np.max(inputs_8x8[i]), ' min:', np.min(inputs_8x8[i]), file=file)
        inputs_8x8[i] = (inputs_8x8[i] - np.min(inputs_8x8[i])) / (np.max(inputs_8x8[i]) - np.min(inputs_8x8[i]))

file.close()

file = open('targetsConversion8x8.txt', 'w')

print("Normalizing 8 by 8 targets")
for i in range(len(targets_8x8)):
    if np.max(targets_8x8[i]) != np.min(targets_8x8[i]):
        print('Run:', str(i), ' max:', np.max(targets_8x8[i]), ' min:', np.min(targets_8x8[i]), file=file)
        targets_8x8[i] = (targets_8x8[i] - np.min(targets_8x8[i])) / (np.max(targets_8x8[i]) - np.min(targets_8x8[i]))

file.close()

np.save('8x8NormedInputs', inputs_8x8)
np.save('8x8NormedTargets', targets_8x8)
print('done')
