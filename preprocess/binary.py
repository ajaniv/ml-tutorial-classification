import numpy as np
from sklearn import preprocessing

input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])

# Binarize data (set feature values to 0 or 1) according to a threshold.
# This technique is helpful when we have prior knowledge of the data.
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(input_data)
print ("Binarized data:")
print (data_binarized)

"""
Binarized data:
[[1. 0. 1. 0.]
 [0. 1. 0. 1.]
 [0. 1. 0. 0.]]
 """