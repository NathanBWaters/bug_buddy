'''
Supervised Learning algorithm for predicting which tests will fail given
a commit.
'''
import keras
import random

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


def predict(commit: Commit):
    '''
    Predicts using the model
    '''
    pass


def get_training_data(repository: Repository):
    '''
    Returns the repository's data to be ready for training
    '''
    commits = repository.commits
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

        percent = len(commits) / float(i)
        if percent < train_percent:
            train_features.append(feature)
            train_labels.append(label)

        elif percent < train_percent + validation_percent:
            validation_features.append(feature)
            validation_labels.append(label)

        else:
            test_features.append(feature)
            test_labels.append(label)

    return (train_features,
            train_labels,
            validation_features,
            validation_labels,
            test_features,
            test_labels)


def commit_to_feature(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the features for
    determining which test failures might occur from the change.

    In this case, it is simply a numpy array with a 1 for each method that was
    changed and a 0 for each method that was not changed.  The array is in
    alphabetical order.
    '''
    return


def commit_to_label(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the labels for
    determining which test failures might occur from the change.

    In this case, it is simply a numpy array with a 1 for each test that failed
    and a 0 for each test that passed.  The array is in alphabetical order.
    '''
    return


