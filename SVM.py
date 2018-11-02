
"""

Handwritten Digit Recognition using SVM
Program by Jayanth Suresh
Program created for Kaggle competition

"""


# Import Libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

start_time = time.time()

# Load training data
train_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\train.csv') 
test_data = pd.read_csv(r'J:\GitHub\Handwritten digit recognition\Handwritten digit recognition\all\test.csv') 

images = train_data.iloc[0:, 1:] 
labels = train_data.iloc[0:, :1] 

train_images = images/255
test_images = np.asarray(test_data)/255

# ML
clf = SVC(gamma='auto')
clf.fit(train_images, labels)
predictions = clf.predict(test_images)

# Store Results
res = {'ImageId' : range(1, len(predictions)+1), 'Label' : predictions}
df = pd.DataFrame(data=res)
df.to_csv('svm.csv', index = False)

print("Total execution time is : %s seconds" % (time.time() - start_time))