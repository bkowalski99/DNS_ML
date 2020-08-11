import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# Notes
# Up and working for real now, current steps
# Found a way to store the data using numpy
# Now opening the stored data and manually splitting to have a saved dataset
# Next Steps
# - Normalize the inputs DONE
# - Run and try to find a good optimizer for the data DONE
# - Use a decent loss function DONE
# - Set up early stopping DONE
# - look into whats causing the memory mention in the training section (BOUNTY)
#       o 2020-07-21 17:08:20.199500: W tensorflow/core/framework/cpu_allocator_impl.cc:81] Allocation of 1207959552
#       exceeds 10% of free system memory.
#       o above is the generated error code, need to search deeper in StackOverflow
# - ensure the system is training correctly (is the number on the left the number of tests ran?)
# - run for longer
# - comment through code

# Notes from meeting
# - train with one section, apply to different cross section


def scheduler(epoch):
    if epoch < 75:
        return 0.0001
    else:
        return 0.00001


# loads in normalized version of data
loadedinputs = np.load('8x8NormedInputs.npy')
loadedtargets = np.load('8x8NormedTargets.npy')

# splits the data into a training section and a testing section
# ideally this would be done on a 70/30 split but I need to see how the method works
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
training_inputs = training_inputs.reshape(16384, 8, 8, 1)
validation_inputs = validation_inputs.reshape(16384, 8, 8, 1)
testing_inputs = testing_inputs.reshape(16384, 8, 8, 1)
training_targets = training_targets.reshape(16384, 64, 1)
validation_targets = validation_targets.reshape(16384, 64, 1)
testing_targets = testing_targets.reshape(16384, 64, 1)

print(training_inputs.shape)
print(training_targets.shape)
# preliminary error checking
if not len(training_inputs) == len(training_targets):
    print("TRAINING DATA SET DOES NOT MATCH")

if not len(testing_inputs) == len(testing_targets):
    print("VALIDATION DATA SET DOES NOT MATCH")

# building tensorflow dataset from normalized data points
train_dataset = tf.data.Dataset.from_tensors((training_inputs, training_targets))
validation_dataset = tf.data.Dataset.from_tensors((validation_inputs, validation_targets))
testing_dataset = tf.data.Dataset.from_tensors((testing_inputs, testing_targets))

# Setting the program to expect numpy doubles
tf.keras.backend.set_floatx('float64')

# setting up the model
# Changelog
# - added 2 dropout layers
# - took out the 2 relu layers following the pooling ops
# - increased number of nodes in 3rd convolution 64 -> 128
model = keras.Sequential([
    layers.Conv2D(64, (2, 2), activation='relu', kernel_initializer=keras.initializers.GlorotNormal(),
                  input_shape=[8, 8, 1]),
    layers.Conv2D(512, (2, 2), activation='relu'),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.5),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(rate=0.5),
    layers.Dense(1024, activation='swish'),
    layers.Dropout(rate=0.5),
    layers.Dense(64)
])

model.build(input_shape=(1, 8, 8, 1))
model.compile(optimizer='adam',
              loss='logcosh',
              metrics=['mae', 'mse'])

# prints model shape
model.summary()

# set up controls for learning rate and early stopping to quit when data is saturated
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# number of cycles
EPOCHS = 300

# Possible changes to improve training
#   - Update batch_size?
#   - Use a validation set from the data
model.fit(train_dataset, callbacks=[callback, early_stop], validation_data=validation_dataset, use_multiprocessing=True,
          epochs=EPOCHS)

loss, test_mae, test_mse = model.evaluate(testing_dataset, verbose=2)

print('\nLoss:', loss)
print('\nMAE:', test_mae)
print('\nMSE: ', test_mse)

# print sample set of model's outputs to see how close it is getting
predictions = model.predict(testing_dataset).flatten()

predictions = predictions.reshape(1048576)
testing_targets = testing_targets.reshape(1048576)
print(testing_targets.shape)
print(predictions.shape)

a = plt.axes(aspect='equal')
plt.scatter(x=testing_targets, y=predictions)
plt.xlabel('Validation Targets')
plt.ylabel('Test Predictions')
lims = [-1, 1]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

predictions = predictions.reshape(16384, 8, 8)
testing_targets = testing_targets.reshape(16384, 8, 8)
testing_inputs = testing_inputs.reshape(16384, 8, 8)

for z in range(300):
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(testing_inputs[z])
    axs[0, 0].set_title('Prediction Input')
    axs[0, 1].imshow(testing_inputs[z])
    axs[0, 1].set_title('Target Input')
    axs[1, 0].imshow(predictions[z])
    axs[1, 0].set_title('Prediction Result')
    axs[1, 1].imshow(testing_targets[z])
    axs[1, 1].set_title('Target Result')
    plt.show()
