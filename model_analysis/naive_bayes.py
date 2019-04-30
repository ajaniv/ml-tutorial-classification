"""
Naive bayes
"""
import os
import csv
import random
import math
# pylint: disable=invalid-name

def load_csv(filename):
    """Load csv file"""
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    for i in range(len(dataset)): # pylint: disable=consider-using-enumerate
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def split_dataset(dataset, split_ratio):
    """split dataset"""
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]

def separate_by_class(dataset):
    """separate by class"""
    separated = {}
    for i in range(len(dataset)): # pylint: disable=consider-using-enumerate
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def calculate_mean(numbers):
    """calculate the mean"""
    return sum(numbers)/float(len(numbers))

def calculate_stdev(numbers):
    """calculate stdev"""
    avg = calculate_mean(numbers)
    variance = sum([pow(x-avg, 2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

def summarize(dataset):
    """summarize"""
    summaries = [(calculate_mean(attribute), calculate_stdev(attribute))
                 for attribute in zip(*dataset)]

    return summaries

def summarize_by_class(dataset):
    """summarize by class"""
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, instances in separated.items():
        summaries[class_value] = summarize(instances)
    return summaries

def calculate_probability(x, mean_value, stdev_value):
    """calculate probability"""
    try:
        divisor = 2*math.pow(stdev_value, 2)
        exponent = math.exp(-(math. pow(x-mean_value, 2)/(divisor)))
    except ZeroDivisionError:
        return 0
    return (1 / (math.sqrt(2*math.pi) * stdev_value)) * exponent

def calculate_class_probabilities(summaries, inputVector):
    """calculate class probabilities"""
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)): # pylint: disable=consider-using-enumerate
            mean_value, stdev_value = class_summaries[i]
            x = inputVector[i]
            probabilities[class_value] *= calculate_probability(x, mean_value, stdev_value)
    return probabilities

def predict(summaries, inputVector):
    """predict"""
    probabilities = calculate_class_probabilities(summaries, inputVector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

def get_predictions(summaries, testSet):
    """get predictions"""
    predictions = []
    for i in range(len(testSet)): # pylint: disable=consider-using-enumerate
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

def get_accuracy(test_set, predictions):
    """get accuracy"""
    correct = 0
    for i in range(len(test_set)): # pylint: disable=consider-using-enumerate
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0

def main():
    """main function"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(dir_path, "data")
    filename = 'pima-indians-diabetes-data.csv'
    split_ratio = 0.67
    dataset = load_csv(os.path.join(data_dir, filename))
    training_set, test_set = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),
                                                                       len(training_set),
                                                                       len(test_set)))

    # prepare model
    summaries = summarize_by_class(training_set)

    # test model
    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}%'.format(accuracy))

main()
