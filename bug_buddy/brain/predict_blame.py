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
    get_blame_counts_for_tests,
    # NUM_INPUT_COMMITS,
    # NUM_FEATURES
)
from bug_buddy.constants import SYNTHETIC_CHANGE
from bug_buddy.db import get, session_manager
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Repository, TestResult

import tensorflow as tf


numpy.set_printoptions(suppress=True, precision=4)
BLAME_PREDICTION_MODEL = 'predict_blame.h5'

test_result_prediction_model = None

# A TfRecord file that stores the vectors.  Used for quick ingestion during
# training and caches the computed output of each test failure.
TF_RECORD_FILE = get_output_dir('train.tfrecords')


def to_tf_float(value):
    '''
    Utility function for wrapping a value as a int 64 for Tensorflow
    '''
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_records(repository: Repository):
    '''
    Writes the training data to a TfRecord file
    '''
    writer = tf.python_io.TFRecordWriter(TF_RECORD_FILE)
    print('Getting commits')
    commits = get_commits(repository, synthetic=True)
    print('Got {} of em'.format(len(commits)))

    for commit in commits:
        print('Doing commit: {}'.format(commit))
        for failed_result in commit.failed_test_results:

            feature = test_failure_to_feature(failed_result)
            label = test_failure_to_label(failed_result)

            feature = {'feature': to_tf_float(feature),
                       'label': to_tf_float(label)}

            # Serialize to string and write to file
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())

    writer.close()


def commit_generator(repository_id: int, batch_size: int):
    '''
    Returns augmented commit data for training
    '''
    with session_manager() as session:
        repository = get(session, Repository, id=repository_id)
        while True:
            features = []
            labels = []

            while len(features) < 1000:
                # commit = get(session, Commit, id=1809)
                commit = get_commits(
                    repository, num_commits=batch_size, synthetic=True)[0]

                if not commit.failed_test_results:
                    continue

                print(commit)
                features.extend(commit_to_features(commit, add_noise=True))
                labels.extend(commit_to_labels(commit))

            print('Yielding batch size of {}'.format(batch_size))
            print('Features size {}'.format(len(features)))
            print('Labels size {}'.format(len(labels)))
            yield numpy.stack(features), numpy.stack(labels)


def tensor_feeder(filename,
                  perform_shuffle=False,
                  repeat_count=1,
                  batch_size=1):
    '''
    Feeds TfRecords data into the Estimator

    @returns: a two-element tuple organized as follows:
    - The first element must be a dictionary in which each input feature is a
      key. We have only one 'dense_1' here which is the input layer name for
      the model.
    - The second element is a list of labels for the training batch.
    '''
    def parse(serialized):
        '''
        Parses the TFRecords and returns them as tf Tensors
        '''
        feature = {
            'feature': tf.FixedLenFeature([], tf.float32),
            'label': tf.FixedLenFeature([], tf.float32)
        }
        example = tf.parse_single_example(serialized, feature)

        return (dict({'dense_1': example['feature']}), example['label'])

    dataset = tf.data.TFRecordDataset(filenames=filename)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(parse)

    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)

    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def train(repository: Repository):
    '''
    Creates and trains a neural network.  It then exports a model.
    '''
    # bringing this into the method because keras takes so long to load

    # importing Keras from Tensorflow is necessary for transforming a Keras
    # model into an Estimator
    import tensorflow as tensorflow
    tensorflow.enable_eager_execution()
    from tensorflow.train import AdamOptimizer
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.callbacks import ModelCheckpoint
    from tensorflow.python.keras.layers import Dense, Activation

    model = Sequential()
    model.add(Dense(1187, input_shape=(1187, )))
    model.add(Activation('relu'))
    model.add(Dense(674))
    model.add(Activation('relu'))
    model.add(Dense(337))
    model.add(Activation('sigmoid'))

    optimizer = AdamOptimizer()
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])

    print(model.summary())

    # Convert our keras model into a Tensorflow Estimator so that we can easily
    # import TfRecords
    # estimator = tf.keras.estimator.model_to_estimator(
    #     keras_model=model, model_dir=get_output_dir())

    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: tensor_feeder(
    #         TF_RECORD_FILE,
    #         perform_shuffle=True,
    #         repeat_count=5,
    #         batch_size=20),
    #     max_steps=500)
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: tensor_feeder(
    #         TF_RECORD_FILE,
    #         perform_shuffle=False,
    #         batch_size=1))

    # estimator.train(input_fn=tensor_feeder(TF_RECORD_FILE))

    batch_size = 1

    generator = commit_generator(repository.id, batch_size)

    checkpoint_weights_filename = get_output_dir('predict_blame_{epoch}.h5f')

    callbacks = [
        ModelCheckpoint(checkpoint_weights_filename, period=10),
    ]

    # Fit the model
    model.fit_generator(
        generator,
        epochs=4000,
        callbacks=callbacks,
        steps_per_epoch=1,
        verbose=1)

    model.save(get_output_dir(BLAME_PREDICTION_MODEL))


def predict_blames(commit: Commit):
    '''
    Predicts the functions to blame for each test failure in the commit's
    latest test run
    '''
    return [predict_blame(test_failure) for test_failure
            in commit.failed_test_results]


def predict_blame(test_failure: TestResult):
    '''
    Predicts using the model
    '''
    from keras.models import load_model
    model = load_model(get_output_dir(BLAME_PREDICTION_MODEL))
    feature = test_failure_to_feature(test_failure)
    prediction_vector = model.predict(numpy.array([feature, ]))
    test_failure.blamed_function_prediction_vector = (
        numpy.around(prediction_vector, decimals=3).tolist()[0])
    test_failure.summary()
    print('\n')
    # for blame in test_failure.blames:
    #     if blame.diff.function == test_failure.blamed_function_prediction[0][0]:
    #         print('GOT IT RIGHT\n')
    #     else:
    #         import pdb; pdb.set_trace()
    #         print('got it wrong..\n')
    return prediction_vector


def validate(repository):
    '''
    Validates a model against a repository
    '''
    from keras.models import load_model
    model = load_model(get_output_dir(BLAME_PREDICTION_MODEL))
    commits, validation_features, validation_labels = get_validation_data(repository)
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

        label = commit_to_blame_matrix_labels(commit)

        validation_features.append(feature)
        validation_labels.append(label)

    return (commits,
            numpy.stack(validation_features),
            numpy.stack(validation_labels))


def commit_to_features(commit: Commit, add_noise: bool=False):
    '''
    Converts a commit to a list of features and labels
    '''
    features = []

    for test_failure in commit.failed_test_results:
        features.append(
            test_failure_to_feature(test_failure, add_noise=add_noise))

    return features


def commit_to_labels(commit: Commit):
    '''
    Converts a commit to a list of features and labels
    '''
    labels = []

    for test_failure in commit.failed_test_results:
        labels.append(test_failure_to_label(test_failure))

    return labels


def test_failure_to_feature(test_failure: TestResult, add_noise=False):
    '''
    Convert a test failure into a feature.  This contains all the features
    for each method in relation to the test
    '''
    # get a vector of all the tests that are failing and signify with 2 which
    # test we are focused on
    test_result_vector = numpy.array([])
    for test_result in test_failure.test_run.test_results_ordered:
        if test_result.id == test_failure.id:
            # we are signifying that this is current test that we are concerned
            # about by setting it to 2.
            test_result_vector = numpy.append(test_result_vector, 2.0)

        elif test_result.failed:
            test_result_vector = numpy.append(test_result_vector, 1.0)

        else:
            test_result_vector = numpy.append(test_result_vector, 0.0)

    # get a vector of all the functions and features of each function and the
    # test failure in question
    functions = test_failure.test_run.commit.functions

    function_feature_vector = numpy.array([])
    blame_count_dict = get_blame_counts_for_tests(test_failure.test)
    for function in functions:
        function_was_altered = any(
            [diff.commit.id == test_failure.test_run.commit.id
             for diff in function.diffs])
        blame_count = blame_count_dict.get(function.id, 0)

        function_feature_vector = numpy.append(
            function_feature_vector,
            numpy.array([function_was_altered, blame_count]))

    if add_noise:
        num_methods_altered = random.randint(0, 10)
        for i in range(num_methods_altered):
            method_to_alter = random.randint(0, len(functions) - 1)
            function_feature_vector[method_to_alter * 2] = 1.0

    return numpy.concatenate((function_feature_vector, test_result_vector))


def test_failure_to_label(test_failure: TestResult):
    '''
    Convert a test failure into a label.  A function has a 1 if it is blamed
    for the test failure
    '''
    functions = test_failure.test_run.commit.functions
    label = numpy.array([])
    blamed_functions = [blame.function for blame in test_failure.blames]
    for function in functions:
        if function in blamed_functions:
            label = numpy.append(label, 1.0)
        else:
            label = numpy.append(label, 0.0)

    return label


def commit_to_blame_matrix_labels(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the labels for
    determining which method is to be blamed for each test failure.

    It is a 1D flattened array of a 2D matrix where the columns are functions
    and the rows are tests.  So the array has the following pattern:
      [ each function info for testA, each function info for testB, ... ]

    '''
    sorted_test_results = commit.latest_test_run.test_results
    sorted_test_results.sort(key=lambda test_result: test_result.test.id)

    sorted_functions = commit.repository.functions
    sorted_functions.sort(key=lambda func: func.id, reverse=False)

    label = numpy.zeros(len(sorted_test_results) * len(sorted_functions))

    for x, test_result in enumerate(sorted_test_results):
        if test_result.failed:
            if not test_result.blames:
                import pdb; pdb.set_trace()

            for blame in test_result.blames:
                try:
                    y = sorted_functions.index(blame.function)
                    index = (x * len(sorted_functions)) + y
                    label[index] = 1.0
                except ValueError:
                    import pdb; pdb.set_trace()
                    print('could not find the function')

                except IndexError:
                    import pdb; pdb.set_trace()
                    print('bad index')

    label = numpy.asarray(label)
    return numpy.split(label, label.size)


