'''
Supervised Learning algorithm for predicting which tests will fail given
a commit.
'''
import numpy
import random
import sys
import tensorflow as tf
import keras_metrics as km

# importing Keras from Tensorflow is necessary for transforming a Keras
# model into an Estimator
import tensorflow as tensorflow
from tensorflow.train import AdamOptimizer
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import (
    Dense,
    Activation,
    BatchNormalization,
    Dropout)
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
from bug_buddy.blaming import get_synthetic_children_commits
from bug_buddy.constants import SYNTHETIC_CHANGE, TEST_OUTPUT_FAILURE
from bug_buddy.db import get, session_manager, Session, get_all
from bug_buddy.logger import logger
from bug_buddy.schema import (
    Commit,
    CommitList,
    Function,
    Repository,
    TestResult)

tensorflow.enable_eager_execution()


numpy.set_printoptions(suppress=True, precision=4, threshold=sys.maxsize)

EXPERIMENT_ID = 5
BLAME_PREDICTION_MODEL_FILE = 'predict_blame_{experiment_id}.h5'.format(
    experiment_id=EXPERIMENT_ID)

# global variable so we only have to import the model once
BLAME_PREDICTION_MODEL = None

# A TfRecord file that stores the vectors.  Used for quick ingestion during
# training and caches the computed output of each test failure.
TF_RECORD_FILE = get_output_dir('exp_2_predict_blame.tfrecords')


def predict_blames(commit: Commit):
    '''
    Predicts the functions to blame for each test failure in the commit's
    latest test run
    '''
    return [predict_blame(test_failure) for test_failure
            in commit.failed_test_results]


def commit_generator(repository_id: int, batch_size: int, no_noise_epochs=200):
    '''
    Returns augmented commit data for training
    '''
    epoch_num = 1
    add_noise = False
    with session_manager() as session:
        repository = get(session, Repository, id=repository_id)
        while True:
            print('Gettin data')
            if epoch_num > no_noise_epochs:
                add_noise = True

            failed_test_results = get_all(
                session,
                TestResult,
                limit=1000,
                random=True,
                repository_id=repository.id,
                status=TEST_OUTPUT_FAILURE)

            features = []
            labels = []
            for failed_result in failed_test_results:
                features.append(
                    numpy.array(
                        test_failure_to_feature(failed_result,
                                                add_noise=add_noise),
                        copy=True))

                labels.append(
                    numpy.array(test_failure_to_label(failed_result),
                                copy=True))

            epoch_num += 1

            print('Yielding data')
            yield numpy.stack(features), numpy.stack(labels)


def train(repository: Repository):
    '''
    Creates and trains a neural network.  It then exports a model.
    '''
    # bringing this into the method because keras takes so long to load
    model = get_model_schema()

    train_model_keras(repository, model)


def train_model_keras(repository, model):
    '''
    Convert our keras model into a Tensorflow Estimator so that we can easily
    import TfRecords
    '''
    batch_size = 1

    generator = commit_generator(repository.id, batch_size)

    checkpoint_weights_filename = get_output_dir(
        'predict_blame_{experiment_id}'.format(experiment_id=EXPERIMENT_ID) +
        '_{epoch}.h5f')

    callbacks = [
        ModelCheckpoint(checkpoint_weights_filename, period=10),
    ]

    # Fit the model
    model.fit_generator(
        generator,
        epochs=800,
        callbacks=callbacks,
        steps_per_epoch=1,
        verbose=1)

    model.save(get_output_dir(BLAME_PREDICTION_MODEL_FILE))


def get_model_schema():
    '''
    Attempt had a flat 1D input made up of two conatenated vectors:
        - First part of the vector was the method data in relation to the test
          in question.
            - function was altered: 0 or 1
            - number of times the function has been synthetically blamed: int

        - Second part of the vector was for each test it had two components:
            - did fail: 0 or 1
            - is test in question: 0 or 1

    Major updates:
        - Using batch normalization
        - Using dropout
        - Using binary crossentropy instead of categorical cross entropy
    '''
    model = Sequential()
    model.add(Dense(1700, input_shape=(1700, )))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(674))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Activation('relu'))

    model.add(Dense(337))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Activation('sigmoid'))

    optimizer = AdamOptimizer()

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy', km.binary_precision(), km.binary_recall()])

    print(model.summary())

    return model


def load_blame_model():
    '''
    Returns the prediction model
    '''
    global BLAME_PREDICTION_MODEL

    if not BLAME_PREDICTION_MODEL:
        logger.info('Loading blame prediction model')
        BLAME_PREDICTION_MODEL = load_model(get_output_dir(
            BLAME_PREDICTION_MODEL_FILE))
        # BLAME_PREDICTION_MODEL.load_weights(
        #     get_output_dir('predict_blame_3_1680.h5f'))

    return BLAME_PREDICTION_MODEL


def predict_blame(test_failure: TestResult):
    '''
    Predicts using the model
    '''
    model = load_blame_model()

    feature = test_failure_to_feature(test_failure)
    prediction_vector = model.predict(numpy.array([feature, ]))

    # store the prediction
    test_failure._blamed_function_prediction = prediction_vector[0]

    return prediction_vector


def test_failure_to_feature(test_failure: TestResult,
                            add_noise=False,
                            use_cache=True):
    '''
    Convert a test failure into a feature.  This contains all the features
    for each method in relation to the test

    @returns: a 1D vector made up of two conatenated 1D vectors:
        - First part of the vector was the method data in relation to the test
          in question.
            - function was altered: 0 or 1
            - number of times the function has been synthetically blamed: int

        - Second part of the vector was for each test it had two components:
            - did fail: 0 or 1
            - is test in question: 0 or 1
    '''
    if use_cache and test_failure.cached_function_blame_feature is not None:
        blame_features = test_failure.cached_function_blame_feature
        if add_noise:
            blame_features = add_test_failure_noise(
                test_failure, blame_features)

        return blame_features

    # get the previous commits of the synthetic commit
    previous_commits = get_synthetic_children_commits(
        test_failure.test_run.commit)
    commit = test_failure.test_run.commit

    # add the current commit to the list of commits to look at in seeing if
    # a function was altered since the test was passing
    previous_commits.append(commit)

    test_result_vector = numpy.array([])

    for test_result in test_failure.test_run.test_results_ordered:
        # test is the test in question: 0 or 1
        if test_result.id == test_failure.id:
            test_result_vector = numpy.append(test_result_vector, 1.0)
        else:
            test_result_vector = numpy.append(test_result_vector, 0.0)

        #  test failed: 0 or 1
        if test_result.failed:
            test_result_vector = numpy.append(test_result_vector, 1.0)
        else:
            test_result_vector = numpy.append(test_result_vector, 0.0)

    # get a vector of all the functions and features of each function and the
    # test failure in question
    functions = test_failure.test_run.commit.functions

    function_feature_vector = numpy.array([])
    blame_count_dict = get_blame_counts_for_tests(test_failure.test)
    for function in functions:
        # whether or not the function was altered in the commit
        was_altered = int(any(
            [diff.commit.id == commit.id for diff in function.diffs]))

        blame_count = blame_count_dict.get(function.id, 0)

        function_feature_vector = numpy.append(
            function_feature_vector,
            numpy.array([was_altered, blame_count]))

    if add_noise:
        function_feature_vector = add_test_failure_noise(
            test_failure, function_feature_vector)

    feature_vector = numpy.concatenate(
        (function_feature_vector, test_result_vector))
    test_failure._cached_function_blame_feature = feature_vector

    return feature_vector


def test_failure_to_label(test_failure: TestResult, use_cache=True):
    '''
    Convert a test failure into a label.  A function has a 1 if it is blamed
    for the test failure
    '''
    if use_cache and test_failure.cached_function_blame_label is not None:
        return test_failure.cached_function_blame_label

    functions = test_failure.test_run.commit.functions
    label = numpy.array([])
    blamed_functions = [blame.function for blame in test_failure.blames]
    for function in functions:
        if function in blamed_functions:
            label = numpy.append(label, 1.0)
        else:
            label = numpy.append(label, 0.0)

    test_failure._cached_function_blame_label = label
    return label


def add_test_failure_noise(test_failure, input_vector):
    '''
    Adds noise to the test failure input vector
    '''
    functions = test_failure.test_run.commit.functions

    num_methods_altered = random.randint(0, 10)
    for i in range(num_methods_altered):
        method_to_alter = random.randint(0, len(functions) - 1)
        input_vector[method_to_alter * 2] = 1.0

    return input_vector


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


def function_changed_since_test_passed(test_failure_result: TestResult,
                                       function: Function,
                                       previous_commits: CommitList=[]) -> bool:
    '''
    Return True if the function was altered since the failing test was passing.
    This counts the current and commit and all commits since he last known
    commit where the test was passing.

    @param commit: the current commit
    @param function: the function that we want to know if it was altered
    @param test: the test that is failing
    @param previous_commits: if previous commits is not specified, then it will
                             keep retrieving commits from the database and a
                             test result for a commit is passing or not.
    @return: bool
    '''
    if previous_commits:
        # make sure the latest commit is first to be tested against
        previous_commits.sort(key=lambda c: c.id, reverse=True)

        for commit in previous_commits:
            # see if the test is passing in this commit
            commit_test_result = [
                test_failure_result.test.id == test_result.test.id for
                test_result in commit.latest_test_run.test_results][0]

            # if the test is passing for this commit, then we know the function
            # was not altered after it was passing.
            if commit_test_result.passed:
                return False

            # check if the function was altered in this commit
            function_altered_in_commit = any(
                [diff.commit.id == commit.id for diff in function.diffs])

            # if the function was altered and the test is not passing yet when
            # looking at previous commits, then return True
            if function_altered_in_commit and commit_test_result.failed:
                return True

        # If we are here, then the test was not altered and the test was failing
        # the whole time?  This shouldn't happen with synthetic commits where
        # the children commits all start from a clean base
        import pdb; pdb.set_trace()
        return False

    else:
        raise NotImplementedError()


def validate(repository):
    '''
    Validates a model against a repository
    '''
    model = load_model(get_output_dir(BLAME_PREDICTION_MODEL_FILE))
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


def cache_test_results(repository: Repository):
    '''
    Writes the training data to a TfRecord file
    '''
    session = Session.object_session(repository)
    print('Getting commits')
    commits = get_commits(repository, synthetic=True)
    print('Got {} of them'.format(len(commits)))

    for commit in commits:
        print('Doing commit: {} with {} failed tests'.format(
            commit, len(commit.failed_test_results)))

        for failed_result in commit.failed_test_results:
            test_failure_to_feature(failed_result)
            test_failure_to_label(failed_result)

        session.commit()


def tensor_feeder(filename, perform_shuffle=False, repeat_count=1, batch_size=1):
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


def train_model_as_estimator(model):
    '''
    Convert our keras model into a Tensorflow Estimator so that we can easily
    import TfRecords
    '''
    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=get_output_dir())

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

    estimator.train(input_fn=tensor_feeder(TF_RECORD_FILE))

