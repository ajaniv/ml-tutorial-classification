"""
Decision Tree using iris dataset
"""
# pylint: disable=invalid-name
# Starting implementation
# %matplotlib inline
# requires install of ipython, graphviz
import os
import pandas as pd
from IPython.display import Image
import pydotplus as pydot
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(dir_path, "data")
dataframe = pd.read_csv(os.path.join(data_dir, "iris_df.csv"))
dataframe.columns = ["X1", "X2", "X3", "X4", "Y"]
print(dataframe.head(5))

#implementation

decision = tree.DecisionTreeClassifier(criterion="gini")
X = dataframe.values[:, 0:4]
Y = dataframe.values[:, 4]
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.3)
decision.fit(trainX, trainY)
print("Accuracy: \n", decision.score(testX, testY))

#Visualisation
dot_data = StringIO()
tree.export_graphviz(decision, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
png = graph.create_png()
Image(png)

plt.show()
tree.export_graphviz(decision, out_file='tree.dot')

"""
Accuracy:
 0.9333333333333333
Warning: Could not load "/usr/local/lib/graphviz/libgvplugin_pango.6.dylib" - file not found
""" # pylint: disable=pointless-string-statement
