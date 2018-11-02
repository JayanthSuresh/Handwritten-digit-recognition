
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
start_time = time.time()

# Load training data
train_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\train.csv') 
test_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\test.csv') 

images = train_data.iloc[0:, 1:] # 5000x784
labels = train_data.iloc[0:, :1] # 5000x1

# Data preprocessing
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
train_images = np.asarray(images)/255.0
test_images = np.asarray(test_data)/255

# ML
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation = tf.nn.relu),
    keras.layers.Dense(10, activation = tf.nn.softmax),
    ])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(train_images.reshape((train_images.shape[0], 28, 28)), labels, epochs = 10)

# Predictions
result = model.predict(test_images.reshape(test_images.shape[0],28,28))
predictions = np.zeros(result.shape[0])
for i in range (result.shape[0]):
    predictions[i] = np.argmax(result[i])
    
# Store Results
res = {'ImageId' : range(1, result.shape[0]+1), 'Label' : predictions}
df = pd.DataFrame(data=res)
df.to_csv('NN.csv', index = False, encoding='utf-8')

print("Total execution time is : %s seconds" % (time.time() - start_time))