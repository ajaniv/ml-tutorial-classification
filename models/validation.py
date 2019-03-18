from sklearn import model_selection


from tutorial_classification.models.config import validation_size, seed


def split(dataset):
    """"Split-out validation dataset"""
    array = dataset.values
    X = array[:,0:4]
    Y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = \
        model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    return X_train, X_validation, Y_train, Y_validation



