'''
Utilities for training our models
'''
from __future__ import division
import os
import numpy
import random
from sqlalchemy import desc
from sqlalchemy.sql.expression import func

from bug_buddy.constants import (
    SYNTHETIC_CHANGE,
    TEST_OUTPUT_SUCCESS,
    TEST_OUTPUT_FAILURE,
    TEST_OUTPUT_NOT_RUN,
    TEST_OUTPUT_SKIPPED)
from bug_buddy.db import Session
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Commit, Blame, Function, Test


# the number of commits that are fed into the neural network
NUM_INPUT_COMMITS = 2

# the locations of the following attributes inside the state tensor
FUNCTION_ALTERED_LOC = 0
TEST_STATUS_LOC = 1
BLAME_COUNT_LOC = 2

# Total number of features
NUM_FEATURES = 3

TEST_STATUS_TO_ID_MAP = {
    TEST_OUTPUT_NOT_RUN: 0,
    TEST_OUTPUT_SKIPPED: 1,
    TEST_OUTPUT_FAILURE: 2,
    TEST_OUTPUT_SUCCESS: 3,
}


def get_output_dir(file_name: str=None):
    '''
    Returns the path to the output directory
    '''
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__)))),
        'model_output')

    # create the directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if file_name:
        output_dir = os.path.join(output_dir, file_name)

    return output_dir


def cache_commits(repository: Repository):
    '''
    Makes sure all commits have been properly converted into a tensor
    '''
    session = Session.object_session(repository)
    synthetic_commits = (
        session.query(Commit)
        .filter(Commit.commit_type == SYNTHETIC_CHANGE)
        .filter(Commit.repository_id == repository.id)
    ).all()

    for commit in synthetic_commits:
        commit_to_tensor(commit)


def get_input_shape(repository: Repository):
    '''
    Returns the shape of the inputs
    '''
    num_functions = len(repository.functions)
    num_tests = len(repository.tests)
    return (NUM_INPUT_COMMITS, num_functions, num_tests, NUM_FEATURES)


def get_test_function_blame_count(function: Function, test: Test):
    '''
    Returns the number of times a function change has been blamed for a test
    '''
    # A blame is made up of diff id and test result id, so given a test and a
    # function means we have to do some expensive queries
    session = Session.object_session(function)
    return (session.query(Blame)
                   .filter(Blame.function_id == function.id)
                   .filter(Blame.test_id == test.id)
                   .count())


def get_blame_counts_for_function(function):
    '''
    Returns the number of times a function change has been blamed for a test
    '''
    # A blame is made up of diff id and test result id, so given a test and a
    # function means we have to do some expensive queries
    session = Session.object_session(function)
    blames = (session.query(Blame.test_id, func.count(Blame.test_id))
                     .filter(Blame.function_id == function.id)
                     .group_by(Blame.test_id)
                     .all())

    blame_dict = {}
    for test_id, blame_count in blames:
        blame_dict[test_id] = blame_count

    return blame_dict


def get_blame_counts_for_tests(test: Test):
    '''
    Returns the number of times a function change has been blamed for a test
    '''
    # A blame is made up of diff id and test result id, so given a test and a
    # function means we have to do some expensive queries
    session = Session.object_session(test)
    blames = (session.query(Blame.function_id, func.count(Blame.function_id))
                     .filter(Blame.test_id == test.id)
                     .group_by(Blame.function_id)
                     .all())

    blame_dict = {}
    for function_id, blame_count in blames:
        blame_dict[function_id] = blame_count

    return blame_dict


def commit_to_tensor(commit):
    '''
    Converts an individual commit into a tensor with the following shape:

                functionA                               functionB
    --------------------------------------------------------------------------
    testA      [function_altered,
                test_status,                               ...
                blame_count]
    --------------------------------------------------------------------------
    testB       ...                                        ...
    --------------------------------------------------------------------------

    or another way of looking at the shape:

    [functionA: [testA: [functionA_altered, testA_status, blame_count],
                [testB: [functionA_altered, testB_status, blame_count],
     ...
    ]
    '''
    if commit._commit_tensor_binary:
        # logger.info('Returning cache for {}'.format(commit))
        return commit.commit_tensor

    logger.info('Creating cached commit tensor for {}'.format(commit))
    session = Session.object_session(commit)

    sorted_functions = commit.repository.functions
    sorted_functions.sort(key=lambda func: func.id, reverse=False)

    sorted_tests = commit.repository.tests
    sorted_tests.sort(key=lambda test: test.id, reverse=False)

    # the current features are:
    #   function_altered
    #   test_status
    #   blame_count
    commit_tensor = numpy.zeros((len(sorted_functions),
                                 len(sorted_tests),
                                 NUM_FEATURES))

    # store the results of the tests for the commit in a dictionary for quick
    # lookup
    commit_results = {}
    if commit.test_runs:
        test_run = commit.test_runs[0]
        for test_result in test_run.test_results:
            commit_results[test_result.test.id] = (
                TEST_STATUS_TO_ID_MAP[test_result.status])

    logger.info('Commit Test Results: {}'.format(commit_results))

    for i in range(len(sorted_functions)):
        function = sorted_functions[i]
        logger.info('On function: {}'.format(function))

        function_was_altered = any([diff.commit.id == commit.id for diff in
                                    function.diffs])
        blame_counts = get_blame_counts_for_function(function)
        for j in range(len(sorted_tests)):
            # Step 1 - add whether or not the function was altered for this
            # commit.  1 for altered, 0 otherwise.
            commit_tensor[i][j][FUNCTION_ALTERED_LOC] = int(function_was_altered)

            # Step 2 - add the status of the test.  If the test is not ran
            # the id will be 0, which represents that the test has not been
            # ran yet
            test = sorted_tests[j]
            commit_tensor[i][j][TEST_STATUS_LOC] = commit_results.get(
                test.id, TEST_STATUS_TO_ID_MAP[TEST_OUTPUT_NOT_RUN])

            # Step 3 - add the blame count, which represents how many times
            # the function has been blamed for the test
            blame_count = blame_counts.get(test.id, 0)
            commit_tensor[i][j][BLAME_COUNT_LOC] = blame_count

    commit._commit_tensor_binary = commit_tensor
    session.commit()
    return commit_tensor


def get_previous_commits(commit: Commit, num_commits: int=NUM_INPUT_COMMITS):
    '''
    Returns the previous commits
    '''
    session = Session.object_session(commit)
    if commit.commit_type == SYNTHETIC_CHANGE:
        return (
            session.query(Commit)
            .filter(Commit.commit_type == SYNTHETIC_CHANGE)
            .filter(Commit.id < commit.id)
            .order_by(desc(Commit.id))
            .limit(num_commits)
        ).all()
    else:
        raise NotImplementedError()


def commit_to_state(commit: Commit,
                    max_length=NUM_INPUT_COMMITS):
    '''
    Converts a commit to a tensor that stores features about the commit and
    its previous commits
    '''
    commits = get_previous_commits(commit, num_commits=max_length - 1)
    commits.append(commit)

    # keep adding the original commit if there are less than max_length commits
    # preceding the commit
    while len(commits) < max_length:
        commits.append(commit)

    # sort by id, so that the latest commit is first
    commits.sort(key=lambda commit: commit.id, reverse=True)

    state_tensor = []
    for commit in commits:
        # used the cached tensor if it exists.  Otherwise, create it
        state_tensor.append(commit_to_tensor(commit))

    state_tensor = numpy.asarray(state_tensor)

    return state_tensor


def set_functions_altered_noise(state_tensor, num_noise=None):
    '''
    With our synthetic tests, we only added 'assert False' to a few functions.
    Adds noise to the tensor form of the commit to make it seem like more
    functions were altered than they actually were.  These altered functions
    simulate non-problematic changes.
    '''
    num_functions = state_tensor.shape[1]

    # add noise to which functions were altered.  At most we only should have
    # 20 other functions that were altered.
    num_noise = num_noise or random.randint(0, min(num_functions - 1, 20))

    for _ in range(num_noise):
        # choose a function to set to 1, meaning it was altered
        func_to_alter = random.randint(0, num_functions - 1)
        state_tensor[0][func_to_alter:, :, FUNCTION_ALTERED_LOC] = 1


def set_tests_not_run_noise(state_tensor, num_noise=None):
    '''
    With our synthetic tests, we already know the status of each test.  In
    practice, given a commit at first we will not know the status of any of the
    tests.  Once we have ran a few tests, then we will have a better idea of
    which tests might have changed in status.  We can simulate these
    possibilities by setting certain amount of tests to the status of 'not run'.
    This simulates what it's like for the network to predict which tests might
    have changed in status if we have already ran none or some of the tests.
    '''
    num_tests = state_tensor.shape[2]

    # Determine how many tests have not been ran
    num_noise = num_noise or random.randint(0, num_tests - 1)

    for _ in range(num_noise):
        # set a test to status 'not run'
        test_to_alter = random.randint(0, num_tests - 1)
        state_tensor[0][:, test_to_alter, TEST_STATUS_LOC] = (
            TEST_STATUS_TO_ID_MAP[TEST_OUTPUT_NOT_RUN])


def get_commits(repository, num_commits=None, synthetic=True):
    '''
    Returns a list of commits randomly retrieved from the database
    '''
    session = Session.object_session(repository)
    query = session.query(Commit).filter(Commit.repository_id == repository.id)

    if synthetic:
        query = query.filter(Commit.commit_type == SYNTHETIC_CHANGE)

    if num_commits:
        return query.order_by(func.random()).limit(num_commits).all()

    else:
        return query.all()

