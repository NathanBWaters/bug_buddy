'''
Supervised Learning algorithm for predicting which tests will fail given
a commit.
'''
from keras.models import Sequential
from keras.layers import Dense
import numpy
import random

from bug_buddy.constants import SYNTHETIC_RESET_CHANGE
from bug_buddy.schema import Commit, Repository


def train(repository: Repository):
    '''
    Creates and trains a neural network.  It then exports a model.
    '''
    (train_features,
     train_labels,
     validation_features,
     validation_labels,
     test_features,
     test_labels) = get_training_data(repository)

    # create model
    model = Sequential()
    model.add(Dense(1024, input_dim=train_features.shape[1], activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(train_labels.shape[1], activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(train_features, train_labels, epochs=150, batch_size=10)

    # evaluate the model
    scores = model.evaluate(validation_features, validation_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def predict(commit: Commit):
    '''
    Predicts using the model
    '''
    pass


def get_training_data(repository: Repository):
    '''
    Returns the repository's data to be ready for training
    '''
    commits = [commit for commit in repository.commits
               if (commit.commit_type != SYNTHETIC_RESET_CHANGE and
                   commit.test_runs)]
    random.shuffle(commits)

    train_features = []
    train_labels = []
    validation_features = []
    validation_labels = []
    test_features = []
    test_labels = []

    train_percent = 0.6
    validation_percent = 0.2

    for i in range(len(commits)):
        commit = commits[i]
        feature = commit_to_feature(commit)
        label = commit_to_label(commit)

        percent = 0 if not i else i / float(len(commits))

        if percent < train_percent:
            train_features.append(feature)
            train_labels.append(label)

        elif percent < train_percent + validation_percent:
            validation_features.append(feature)
            validation_labels.append(label)

        else:
            test_features.append(feature)
            test_labels.append(label)

    return (numpy.asarray(train_features),
            numpy.asarray(train_labels),
            numpy.asarray(validation_features),
            numpy.asarray(validation_labels),
            numpy.asarray(test_features),
            numpy.asarray(test_labels))


def commit_to_feature(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the features for
    determining which test failures might occur from the change.

    In this case, it is simply a numpy array with a 1 for each method that was
    changed and a 0 for each method that was not changed.  The array is in
    alphabetical order.
    '''
    commit_diffs = commit.diffs

    # TODO - this doesn't make sense when a commit could have different
    # functions (i.e. adding functions, etc)
    sorted_functions = commit.repository.functions
    sorted_functions.sort(key=lambda func: func.id, reverse=False)

    feature = []
    for function in sorted_functions:
        if function.synthetic_diff in commit_diffs:
            feature.append(1)
        else:
            feature.append(0)

    if len(feature) != 333:
        import pdb; pdb.set_trace()
    return numpy.asarray(feature)


def commit_to_label(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the labels for
    determining which test failures might occur from the change.

    In this case, it is simply a numpy array with a 1 for each test that failed
    and a 0 for each test that passed.  The array is in alphabetical order.
    '''
    sorted_test_results = commit.test_runs[0].test_results
    sorted_test_results.sort(key=lambda test_result: test_result.test.id)

    labels = []
    for test_result in sorted_test_results:
        labels.append(1 if test_result.failed else 0)

    if len(labels) != 513:
        import pdb; pdb.set_trace()
    return numpy.asarray(labels)


