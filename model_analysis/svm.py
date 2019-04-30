"""
SVM model
"""
# pylint: disable=invalid-name
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "data")
df = pd.read_csv(os.path.join(data_dir, 'iris_df.csv'))

df.columns = ['X4', 'X3', 'X1', 'X2', 'Y']
df = df.drop(['X4', 'X3'], 1)
print(df.head())

support = svm.SVC()
X = df.values[:, 0:2]
Y = df.values[:, 2]
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3)
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')
sns.lmplot('X1', 'X2', scatter=True, fit_reg=False, data=df, hue='Y')
plt.ylabel('X2')
plt.xlabel('X1')
plt.show()
