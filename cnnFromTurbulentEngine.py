
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
#       A: Occurs when too many apps are running alongside the IDE
# - ensure the system is training correctly
# - comment through code


def scheduler(epoch):
    if epoch < 150:
        return 0.0001
    else:
        return 0.000001


# Changelog
# - increased number of nodes in network
class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        initializer = 'he_uniform'
        self.conv1_1 = layers.Conv2D(32, 1, activation='relu', kernel_initializer=initializer,
                                     data_format='channels_last', input_shape=[32, 32, 1])
        self.conv1_2 = layers.Conv2D(32, 1, activation='relu', kernel_initializer=initializer)
        # cropped version to be concatenated later
        # had to set cropping tuple to be (0, 0) for the code to compile
        self.crop1 = layers.Cropping2D(cropping=(0, 0))

        # pool for downsampling
        self.pool1 = layers.MaxPooling2D(2, 2)

        # second layer of convolutions
        self.conv2_1 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)
        self.conv2_2 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)

        # had to set cropping tuple to be (0, 0) for the code to compile
        self.crop2 = layers.Cropping2D(cropping=(0, 0))

        # second pool for downsampling
        self.pool2 = layers.MaxPooling2D(2, 2, padding='valid')

        # third layer of convolutions
        self.conv3_1 = layers.Conv2D(128, 1, activation='relu', kernel_initializer=initializer)
        self.conv3_2 = layers.Conv2D(128, 1, activation='relu', kernel_initializer=initializer)

        # had to set cropping tuple to be (0, 0) for the code to compile
        self.crop3 = layers.Cropping2D(cropping=(0, 0))

        # third pool for downsampling
        self.pool3 = layers.MaxPooling2D(2, 2, padding='valid')

        # fourth layer of convolutions
        self.conv4_1 = layers.Conv2D(256, 1, activation='relu', kernel_initializer=initializer)
        self.conv4_2 = layers.Conv2D(256, 1, activation='relu', kernel_initializer=initializer)

        # uses a dropout layer for more robust training
        self.drop1 = layers.Dropout(rate=0.5)

        # final crop
        # had to set cropping tuple to be (0, 0) for the code to compile
        self.crop4 = layers.Cropping2D(cropping=(0, 0))

        self.pool4 = layers.MaxPooling2D(2, 2)

        # fifth (and lowest) layer of convolutions
        self.conv5_1 = layers.Conv2D(512, 1, activation='relu', kernel_initializer=initializer)
        self.conv5_2 = layers.Conv2D(512, 1, activation='relu', kernel_initializer=initializer)
        self.drop2 = layers.Dropout(rate=0.5)

        # first upsampling
        self.up6_1 = layers.UpSampling2D(size=(2, 2))

        self.up6 = layers.Conv2D(256, 1, padding='same', activation='relu', kernel_initializer=initializer)

        # concatenate the upsampled version with the cropped version from the opposite side
        # took out the merge block here, makes more sense to do that in the build method

        self.conv6_1 = layers.Conv2D(256, 1, activation='relu', kernel_initializer=initializer)
        self.conv6_2 = layers.Conv2D(256, 1, activation='relu', kernel_initializer=initializer)

        self.up7_1 = layers.UpSampling2D(size=(2, 2))
        self.up7 = layers.Conv2D(128, 1, padding='same', activation='relu', kernel_initializer=initializer)

        # took out merge block

        self.conv7_1 = layers.Conv2D(128, 1, activation='relu', kernel_initializer=initializer)
        self.conv7_2 = layers.Conv2D(128, 1, activation='relu', kernel_initializer=initializer)

        self.up8_1 = layers.UpSampling2D(size=(2, 2))
        self.up8 = layers.Conv2D(64, 1, padding='same', activation='relu', kernel_initializer=initializer)

        self.conv8_1 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)
        self.conv8_2 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)

        self.up9_1 = layers.UpSampling2D(size=(2, 2))
        self.up9 = layers.Conv2D(64, 1, padding='same', activation='relu', kernel_initializer=initializer)

        self.conv9_1 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)
        self.conv9_2 = layers.Conv2D(64, 1, activation='relu', kernel_initializer=initializer)
        self.conv9_3 = layers.Conv2D(64, 1, padding='same', activation='relu', kernel_initializer=initializer)
        self.conv10 = layers.Conv2D(64, 1, kernel_initializer=initializer)

        self.flattenLayer = tf.keras.layers.Flatten()
        self.denseLayer1 = tf.keras.layers.Dense(128, activation='relu')
        self.denseLayer2 = tf.keras.layers.Dense(256, activation='sigmoid')
        self.drop3 = tf.keras.layers.Dropout(rate=0.5)
        # trying swish?
        self.denseLayer3 = tf.keras.layers.Dense(256, activation='swish')
        self.finalLayer = tf.keras.layers.Dense(1024)

    # Changed the training function to true
    def call(self, inputs, training=True, mask=None):
        conv1_1 = self.conv1_1(inputs)
        conv1_2 = self.conv1_2(conv1_1)
        crop1 = self.crop1(conv1_2)

        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        crop2 = self.crop2(conv2_2)

        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        crop3 = self.crop3(conv3_2)

        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        drop1 = self.drop1(conv4_2)
        crop4 = self.crop4(drop1)

        pool4 = self.pool4(drop1)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        drop2 = self.drop2(conv5_2)

        up6_1 = self.up6_1(drop2)
        up6 = self.up6(up6_1)
        merge6 = tf.concat(axis=3, values=[crop4, up6])

        conv6_1 = self.conv6_1(merge6)
        conv6_2 = self.conv6_2(conv6_1)

        up7_1 = self.up7_1(conv6_2)
        up7 = self.up7(up7_1)
        merge7 = tf.concat(axis=3, values=[crop3, up7])

        conv7_1 = self.conv7_1(merge7)
        conv7_2 = self.conv7_2(conv7_1)

        up8_1 = self.up8_1(conv7_2)
        up8 = self.up8(up8_1)
        merge8 = tf.concat(axis=3, values=[crop2, up8])

        conv8_1 = self.conv8_1(merge8)
        conv8_2 = self.conv8_2(conv8_1)

        up9_1 = self.up9_1(conv8_2)
        up9 = self.up9(up9_1)
        merge9 = tf.concat(axis=3, values=[crop1, up9])

        conv9_1 = self.conv9_1(merge9)
        conv9_2 = self.conv9_2(conv9_1)
        conv9_3 = self.conv9_3(conv9_2)
        conv10 = self.conv10(conv9_3)

        flattenlayer = self.flattenLayer(conv10)
        denselayer1 = self.denseLayer1(flattenlayer)
        denselayer2 = self.denseLayer2(denselayer1)
        drop3 = self.drop3(denselayer2)
        denselayer3 = self.denseLayer3(drop3)
        finallayer = self.finalLayer(denselayer3)

        return finallayer

    def model(self):
        x = keras.Input(shape=(32, 32, 1))
        return model(inputs=[x], outputs=self.call(x))

    @property
    def prediction(self):
        return self._prediction

    @property
    def optimize(self):
        return self._optimize

    @property
    def error(self):
        return self._error


# loads in normalized version of data
loadedinputs = np.load('normedInputs.npy')
loadedtargets = np.load('normedTargets.npy')

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
training_inputs = training_inputs.reshape(1024, 32, 32, 1)
validation_inputs = validation_inputs.reshape(1024, 32, 32, 1)
testing_inputs = testing_inputs.reshape(1024, 32, 32, 1)
training_targets = training_targets.reshape(1024, 1024, 1)
validation_targets = validation_targets.reshape(1024, 1024, 1)
testing_targets = testing_targets.reshape(1024, 1024, 1)

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
# splits the data into a training section and a testing section

# Setting the program to expect numpy doubles
tf.keras.backend.set_floatx('float64')

# setting up the model
model = MyModel()
model.build(input_shape=(1024, 32, 32, 1))
# Changelog
# - changed loss from logcosh to mse - changed it back lmao
model.compile(optimizer='adam',
              loss='logcosh',
              metrics=['mae', 'mse'])

# prints model shape
model.summary()

# set up controls for learning rate and early stopping to quit when data is saturated
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=5)

# number of cycles
EPOCHS = 300

# Possible changes to improve training
#   - Update batch_size?
#   - Use a validation set from the data
model.fit(testing_dataset, callbacks=[callback, early_stop], validation_data=validation_dataset, batch_size=400,
          use_multiprocessing=True, epochs=EPOCHS)

loss, test_mae, test_mse = model.evaluate(testing_dataset, verbose=2)

print('\nLoss:', loss)
print('\nMAE:', test_mae)
print('\nMSE: ', test_mse)

# print sample set of model's outputs to see how close it is getting
predictions = model.predict(testing_dataset)
predictions = predictions.reshape(1048576)
testing_targets = testing_targets.reshape(1048576)
print(predictions.shape)
print(testing_targets.shape)

a = plt.axes(aspect='equal')
plt.scatter(x=testing_targets, y=predictions)
plt.xlabel('Validation Targets')
plt.ylabel('Test Predictions')
lims = [-1, 1]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

predictions = predictions.reshape(1024, 32, 32)
testing_targets = testing_targets.reshape(1024, 32, 32)
# learned a bit more about the shapes its predicting, look into noise reduction?
for z in range(30):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(predictions[500+z])
    ax1.set_title('Prediction')
    ax2.imshow(testing_targets[500+z])
    ax2.set_title('Target')
    plt.show()
