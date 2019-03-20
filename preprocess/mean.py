import numpy as np
from sklearn import preprocessing

input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])

# It involves removing the mean from each feature so that it is centered on zero
data_standardized = preprocessing.scale(input_data)
print ("\nMean = ", data_standardized.mean(axis = 0))
print ("Std deviation = ", data_standardized.std(axis = 0))

"""
Mean =  [ 5.55111512e-17 -3.70074342e-17  0.00000000e+00 -1.85037171e-17]
Std deviation =  [1. 1. 1. 1.]

"""