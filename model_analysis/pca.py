"""
PCA model
"""
# pylint: disable=invalid-name
import os
from sklearn import decomposition
from sklearn.model_selection import train_test_split
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "data")
df = pd.read_csv(os.path.join(data_dir, 'iris_df.csv'))
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
print(df.head(5))
pca = decomposition.PCA()
fa = decomposition.FactorAnalysis()
X = df.values[:, 0:4]
Y = df.values[:, 4]
train, test = train_test_split(X, test_size=0.3)
train_reduced = pca.fit_transform(train)
test_reduced = pca.transform(test)

print(pca.n_components_)
