
import numpy as np

import matplotlib.pyplot as plt


# Used to see a heatmap output of one of the xy planes
def heatmap(data, numberofplots):
    for z in range(numberofplots):
        plt.imshow(data[z], interpolation='nearest')
        plt.show()


# Used to compare the smooth field to the full resolution
def heatmapcomparison(data1, data2, numberofplots):
    for z in range(numberofplots):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(data1[z])
        ax1.set_title('256 x 256')
        ax2.imshow(data2[z])
        ax2.set_title('32 x 32')
        plt.show()


# Used to compare the smooth field to the full resolution
def allcompared(data1, data2, data3, numberofplots):
    for z in range(numberofplots):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle("Graph #" + str(z))
        ax1.imshow(data1[z])
        ax1.set_title('h')
        ax2.imshow(data2[z])
        ax2.set_title('hbar')
        ax3.imshow(data3[z])
        ax3.set_title('variance field')
        plt.show()


def makepath(loc, tm, num):
    return loc + '/' + tm + '.00000' + str(num)


def genelem(data, window_size, row, col):
    elem = 0
    for y in range(window_size):
        temp = 0
        for z in range(window_size):
            temp = temp + data[row + z][col + y]
        elem = elem + temp * (1.0 / pow(window_size, 2))
    return elem


def smoother(data, output, window_size):

    for r in range(int(len(data)/window_size)):
        for y in range(int(len(data)/window_size)):
            output[r][y] = genelem(data, window_size, r*window_size, y*window_size)


def variancefield(h, hbr, outputs):

    for z in range(32):
        for y in range(32):
            temp = 0
            for w in range(8):
                for q in range(8):
                    temp = temp + pow((hbr[z][y]-(h[z*8 + w][y*8 + q])), 2)
            outputs[z][y] = temp/64


# NOTES FOR PROGRESS
# This could be changed from a run through then a second run through continuously to being a do-while loop but the code
# would be much less efficient do to added if statements near the end

# GOALS
# Reconstruct the V handler to save all elements from v
#

# QUESTIONS

# Read first set of floats
path = input('Please type the path to your data (do not add closing \'/\'): ')
term = input('And what term are you using? ')
filepath = makepath(path, term, 1)
# Here the arrays are initialized to be filled later
training_inputs = np.array((), float)
training_targets = np.array((), float)

# here the data splits as the V file has more data points than the others
print("reading data 0")
if term is 'V':
    # opens binary file for reading
    file = open(filepath, 'rb')
    # scans to skip over the ensight documentation info that isn't need
    file.seek(244, 0)
    floats = np.fromfile(file, 'f4')
    file.close()

    # this section is special for V, it makes sure all points are being read
    x = np.array_split(floats, 3)
    x[0] = np.reshape(x[0], (256, 256, 256))
    x[1] = np.reshape(x[1], (256, 256, 256))
    x[2] = np.reshape(x[2], (256, 256, 256))

    reshapedFloats = np.array(x[0])
    reshapedFloats = np.concatenate((reshapedFloats, x[1]))
    reshapedFloats = np.concatenate((reshapedFloats, x[2]))

else:

    # opens binary file for reading
    file = open(filepath, 'rb')
    # scans to skip over the ensight documentation info that isn't need
    file.seek(244, 0)
    floats = np.fromfile(file, 'f4')
    file.close()

    reshapedFloats = floats.reshape(256, 256, 256)

# here the smooth field is being generated to create the 32 x 32 planes
hbar = np.zeros((len(reshapedFloats), 32, 32), float)
for i in range(len(hbar)):
    smoother(reshapedFloats[i], hbar[i], 8)

# generates the variance field from the different planes
varouts = np.zeros((len(reshapedFloats), 32, 32), float)
for i in range(len(varouts)):
    variancefield(reshapedFloats[i], hbar[i], varouts[i])

# these are the pieces that will be saved later
training_inputs = hbar
training_targets = varouts

# the rest of the data is read in with 2 for loops, one for getting the single digits, and one for the double digits
for j in range(8):
    number = j + 1
    print("reading data ", number)



    if term is 'V':
        filepath = makepath(path, term, str(number))
        file = open(filepath, 'rb')
        file.seek(244, 0)
        floats = np.fromfile(file, 'f4')
        file.close()

        x = np.array_split(floats, 3)
        x[0] = np.reshape(x[0], (256, 256, 256))
        x[1] = np.reshape(x[1], (256, 256, 256))
        x[2] = np.reshape(x[2], (256, 256, 256))

        reshapedFloats = np.array(x[0])
        reshapedFloats = np.concatenate((reshapedFloats, x[1]))
        reshapedFloats = np.concatenate((reshapedFloats, x[2]))
    else:
        filepath = makepath(path, term, str(number))
        file = open(filepath, 'rb')
        file.seek(244, 0)
        floats = np.fromfile(file, 'f4')
        file.close()
        reshapedFloats = floats.reshape(256, 256, 256)

    # here the smooth field is being generated to create the 32 x 32 planes
    hbar = np.zeros((len(reshapedFloats), 32, 32), float)
    for i in range(len(hbar)):
        smoother(reshapedFloats[i], hbar[i], 8)

    # generates the variance field from the different planes
    varouts = np.zeros((len(reshapedFloats), 32, 32), float)
    for i in range(len(varouts)):
        variancefield(reshapedFloats[i], hbar[i], varouts[i])

    # combine both sets of data
    training_inputs = np.concatenate((training_inputs, hbar))
    training_targets = np.concatenate((training_targets, varouts), axis=1)

# take care of files 10 11 and 12
for j in range(3):
    print("reading data ", 10+j)
    number = int(str('1' + str(j)))
    filepath = path + '/' + term + '.0000' + str(number)

    if term is 'V':
        file = open(filepath, 'rb')
        file.seek(244, 0)
        floats = np.fromfile(file, 'f4')
        file.close()

        x = np.array_split(floats, 3)
        reshapedFloats = np.reshape(x[0], (256, 256, 256))
        reshapedFloats = np.concatenate((reshapedFloats, np.reshape(x[1], (256, 256, 256))))
        reshapedFloats = np.concatenate((reshapedFloats, np.reshape(x[2], (256, 256, 256))))
    else:

        file = open(filepath, 'rb')
        file.seek(244, 0)
        floats = np.fromfile(file, 'f4')
        file.close()

        reshapedFloats = floats.reshape(256, 256, 256)

    # here the smooth field is being generated to create the 32 x 32 planes
    hbar = np.zeros((len(reshapedFloats), 32, 32), float)
    for i in range(len(hbar)):
        smoother(reshapedFloats[i], hbar[i], 8)

    # generates the variance field from the different planes
    varouts = np.zeros((len(reshapedFloats), 32, 32), float)
    for i in range(len(varouts)):
        variancefield(reshapedFloats[i], hbar[i], varouts[i])

    # combine both sets of data
    training_inputs = np.concatenate((training_inputs, hbar))
    training_targets = np.concatenate((training_targets, varouts), axis=1)
    stored = reshapedFloats

np.save('inputs', training_inputs)
np.save('targets', training_targets)
