# Load dataset
import pandas
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']


def load_data():
    dataset = pandas.read_csv(url, names=names)
    return dataset