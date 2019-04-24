#Starting implementation
# %matplotlib inline
# requires install of ipython, graphviz
import pandas as pd

from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus as pydot
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/iris_df.csv")
df.columns = ["X1", "X2", "X3","X4", "Y"]
df.head()

#implementation

decision = tree.DecisionTreeClassifier(criterion="gini")
X = df.values[:, 0:4]
Y = df.values[:, 4]
trainX, testX, trainY, testY = train_test_split( X, Y, test_size = 0.3)
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
"""