"""Random Forest is a popular supervised ensemble learning algorithm. 
‘Ensemble’ means that it takes a bunch of ‘weak learners’ 
and has them work together to form one strong predictor.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('data/iris_df.csv')
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y']
df.head()

forest = RandomForestClassifier(n_estimators=100)
X = df.values[:, 0:4]
Y = df.values[:, 4]
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
forest.fit(trainX, trainY)

print('Accuracy: \n', forest.score(testX, testY))
pred = forest.predict(testX)