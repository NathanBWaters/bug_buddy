'''
Supervised Learning algorithm for predicting which tests will fail given
a commit.
'''
import numpy
import random

from bug_buddy.brain.utils import (
    commit_to_state,
    get_commits,
    get_output_dir,
    set_functions_altered_noise,
    set_tests_not_run_noise,
    NUM_INPUT_COMMITS)
from bug_buddy.constants import SYNTHETIC_CHANGE
from bug_buddy.db import get, session_manager
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Repository


numpy.set_printoptions(suppress=True)
TEST_PREDICTION_MODEL = 'predict_tests.h5'

test_result_prediction_model = None


def commit_generator(repository_id: int, batch_size: int):
    '''
    Returns augmented commit data for training
    '''
    with session_manager() as session:
        repository = get(session, Repository, id=repository_id)
        while True:
            features = []
            labels = []
            commits = get_commits(
                repository, num_commits=batch_size, synthetic=True)

            for commit in commits:
                feature = commit_to_state(commit)
                set_functions_altered_noise(feature)
                set_tests_not_run_noise(feature)

                label = commit_to_test_failure_label(commit)

                features.append(feature)
                labels.append(label)

            yield numpy.stack(features), numpy.stack(labels)


def train(repository: Repository):
    '''
    Creates and trains a neural network.  It then exports a model.
    '''
    # bringing this into the method because keras takes so long to load
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Activation, Convolution3D
    from keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(Convolution3D(128,
                            (2, 4, 4),
                            strides=(2, 3, 3),
                            # input_shape=get_input_shape(repository)))
                            input_shape=(NUM_INPUT_COMMITS, 337, 513, 3)))
    model.add(Activation('relu'))
    model.add(Convolution3D(64, (1, 3, 3), strides=(1, 2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution3D(32, (1, 1, 1), strides=(1, 1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(513))  # 513 is the number of tests
    model.add(Activation('sigmoid'))

    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    print(model.summary())

    batch_size = 10

    generator = commit_generator(repository.id, batch_size)

    checkpoint_weights_filename = get_output_dir('predict_weights_{epoch}.h5f')

    callbacks = [
        ModelCheckpoint(checkpoint_weights_filename, period=3),
    ]

    # Fit the model
    model.fit_generator(
        generator,
        callbacks=callbacks,
        epochs=40,
        steps_per_epoch=batch_size)

    model.save(get_output_dir(TEST_PREDICTION_MODEL))

    validate(repository)


def predict_test_output(commit: Commit):
    '''
    Predicts using the model
    '''
    from keras.models import load_model
    model = load_model(get_output_dir(TEST_PREDICTION_MODEL))
    feature = commit_to_state(commit)
    prediction_vector = model.predict(numpy.array([feature, ]))
    commit.test_result_prediction_data = prediction_vector[0].tolist()
    print('predictions: ', commit.test_result_prediction)


def validate(repository):
    '''
    Validates a model against a repository
    '''
    from keras.models import load_model
    model = load_model(get_output_dir(TEST_PREDICTION_MODEL))
    commits, validation_features, validation_labels = get_validation_data(repository)
    import pdb; pdb.set_trace()
    scores = model.evaluate(validation_features, validation_labels)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


def get_validation_data(repository: Repository):
    '''
    Returns the repository's data to be ready for training
    '''
    commits = []
    i = 0
    logger.info('Retrieving training data')
    for commit in repository.commits:
        if i > 30:
            logger.info('Got 30 commits')
            break

        if (commit.commit_type == SYNTHETIC_CHANGE and commit.test_runs):
            commits.append(commit)
            i += 1

    random.shuffle(commits)

    validation_features = []
    validation_labels = []

    for commit in commits:
        feature = commit_to_state(commit)
        set_functions_altered_noise(feature)
        set_tests_not_run_noise(feature)

        label = commit_to_test_failure_label(commit)

        validation_features.append(feature)
        validation_labels.append(label)

    return (commits,
            numpy.stack(validation_features),
            numpy.stack(validation_labels))


def commit_to_test_failure_label(commit: Commit):
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


