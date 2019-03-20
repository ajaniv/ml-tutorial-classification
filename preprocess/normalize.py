import numpy as np
from sklearn import preprocessing

input_data = np.array([[3, -1.5, 3, -6.4], [0, 3, -1.3, 4.1], [1, 2.3, -2.9, -4.3]])
# Here, the values of a feature vector are adjusted so that they sum up to 1
data_normalized = preprocessing.normalize(input_data, norm  = 'l1')
print ("L1 normalized data:")
print(data_normalized)

"""
L1 normalized data:
[[ 0.21582734 -0.10791367  0.21582734 -0.46043165]
 [ 0.          0.35714286 -0.1547619   0.48809524]
 [ 0.0952381   0.21904762 -0.27619048 -0.40952381]]

"""