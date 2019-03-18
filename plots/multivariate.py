from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from tutorial_classification.data import load_data

dataset = load_data()


# scatter plot matrix
scatter_matrix(dataset)
plt.show()