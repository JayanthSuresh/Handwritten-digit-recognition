
"""

Handwritten Digit Recognition using NN
Program by Jayanth Suresh
Program created for Kaggle competition

"""


# Import Libraries
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

print(tf.__version__)

# Load training data
sample_size = 5000
train_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\train.csv') # 42000x785
test_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\test.csv') # 28000x784

images = train_data.iloc[0:sample_size, 1:] # 5000x784
labels = train_data.iloc[0:sample_size, :1] # 5000x1

# Data preprocessing
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_images = images/255.0
test_images = np.asarray(test_data)/255

# ML
model = keras.Sequential([
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax),
    ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images, labels, epochs = 5)

# Predictions
result = model.predicit(test_images)
predictions = np.zeros(result.shape[0])
for i in range (result.shape[0]):
    predictions[i] = np.argmax[result[i]]
    break

# Store Results
res = {'ImageId' : range(1, result.shape[0]+1), 'Label' : predictions}
df = pd.DataFrame(data=res)
df.to_csv('NN.csv', index = False)

print("Total execution time is : %s seconds" % (time.time() - start_time))