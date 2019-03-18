import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


from tutorial_classification.models.config import seed
from tutorial_classification.data import load_data
from tutorial_classification.models.validation import split

# Test options and evaluation metric
scoring = 'accuracy'


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
names = [name for name, _ in models]


def spot_check(X_train, Y_train):
    """ Spot Check Algorithms """

    # evaluate each model in turn
    results = []
    
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    
    return results

def plot(results):
    """Compare Algorithms"""
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

def predict(X_train, Y_train, X_validation, Y_validation):
    """Make predictions on validation dataset"""
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    
    # In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly 
    # match the corresponding set of labels in y_true.
    print('accuracy_score:', accuracy_score(Y_validation, predictions))
    
    # Compute confusion matrix to evaluate the accuracy of a classification
    print ('confusion_matrix:')
    print(confusion_matrix(Y_validation, predictions))

    """
    The precision is the ratio tp / (tp + fp) where tp is the number of 
    true positives and fp the number of false positives. 
    The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

    The recall is the ratio tp / (tp + fn) where tp is the
    number of true positives and fn the number of false negatives. 
    The recall is intuitively the ability of the classifier to find all the positive samples.
    
    The support is the number of occurrences of each class in y_true.
    """
    print ('classification_report:')
    print(classification_report(Y_validation, predictions))
    
dataset = load_data()
X_train, X_validation, Y_train, Y_validation = split(dataset)

results = spot_check(X_train, Y_train)
predict(X_train, Y_train, X_validation, Y_validation)
plot(results)