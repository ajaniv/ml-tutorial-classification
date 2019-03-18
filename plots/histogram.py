
import matplotlib.pyplot as plt
from tutorial_classification.data import load_data

dataset = load_data()

# histograms
dataset.hist()
plt.show()