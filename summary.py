from tutorial.data import load_data

dataset = load_data()

# shape
print('dataset.shape:')
print(dataset.shape)

# head
print()
print('dataset.head(5):')
print(dataset.head(5))


# statistical summary
# This includes the count, mean, the min and max values as well as some percentiles.
print()
print('dataset.describe():')
print(dataset.describe())

# class distribution
# Let’s now take a look at the number of instances (rows) 
# that belong to each class. We can view this as an absolute count.
print()
print('dataset.groupby(\'class\').size():')
print(dataset.groupby('class').size())