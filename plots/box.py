# box and whisker plots
import matplotlib.pyplot as plt
from tutorial_classification.data import load_data


dataset = load_data()
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()