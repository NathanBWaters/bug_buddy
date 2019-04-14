'''
This is the core class for performing operations on a code repository such as:
    1) Running a test
    2) Assigning blame
    3) Binary searching through a code repository to uncover bugs

'''
from __future__ import division
import numpy as np
import gym
from gym import Env, spaces
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution3D, Input
from keras.optimizers import Adam
import keras.backend as K
import numpy
import random
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from sqlalchemy import desc
from sqlalchemy.sql.expression import func

from bug_buddy.constants import (
    SYNTHETIC_CHANGE,
    TEST_OUTPUT_SUCCESS,
    TEST_OUTPUT_FAILURE,
    TEST_OUTPUT_NOT_RUN,
    TEST_OUTPUT_SKIPPED)
from bug_buddy.db import Session, create
from bug_buddy.logger import logger
from bug_buddy.schema import (
    Repository,
    Commit,
    Blame,
    TestRun,
    TestResult,
    Test)
from bug_buddy.runner import run_test, get_list_of_tests


# the number of commits that are fed into the neural network
NUM_INPUT_COMMITS = 4

# the locations of the following attributes inside the state tensor
FUNCTION_ALTERED_LOC = 0
TEST_STATUS_LOC = 1
BLAME_COUNT_LOC = 2

TEST_STATUS_TO_ID_MAP = {
    TEST_OUTPUT_NOT_RUN: 0,
    TEST_OUTPUT_SKIPPED: 1,
    TEST_OUTPUT_FAILURE: 2,
    TEST_OUTPUT_SUCCESS: 3,
}


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
    num_features = 3
    return (NUM_INPUT_COMMITS, num_functions, num_tests, num_features)


def synthetic_train(repository: Repository):
    '''
    Trains the agent on synthetic data generation
    '''
    logger.info('Training on synthetic data for: {}'.format(repository))
    # logger.info('Caching commit tensors')
    # cache_commits(repository)

    logger.info('Initializing environment')
    brain = Brain(repository)
    env = BugBuddyEnvironment(repository, synthetic_training=True)
    env.reset()
    brain.train(env)


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
        logger.info('Returning cache for {}'.format(commit))
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
    num_function_test_features = 3
    commit_tensor = numpy.zeros((len(sorted_functions),
                                 len(sorted_tests),
                                 num_function_test_features))

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
            .limit(4)
        ).all()
    else:
        raise NotImplementedError()


def commit_to_state(commit: Commit,
                    synthetic_training=False,
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

    if synthetic_training:
        _prepare_state_for_synthetic_training(commit, state_tensor)

    return numpy.asarray(state_tensor)


def _prepare_state_for_synthetic_training(commit, state_tensor):
    '''
    Prepares a state tensor for synthetic training by performing the following
    operations:

        1) Sets all tests to status 'not run'
        2) Adds noise to which functions were altered.  In synthetic training,
           we only added 'assert False' to functions.  In real life, developers
           do not add breaking changes to every function they touch (hopefully).
           So mimic this, we will update the tensor so that it seems other
           methods were also changed
    '''
    logger.info('Adding noise and setting tests results to not run')
    # set all tests to status 'not run'
    state_tensor[0][:, :, TEST_STATUS_LOC] = (
        TEST_STATUS_TO_ID_MAP[TEST_OUTPUT_NOT_RUN])

    # add noise to which functions were altered
    num_noise = random.randint(0, min(15, commit.num_functions))

    for _ in range(num_noise):
        # choose a function to set to 1, meaning it was altered
        func_to_alter = random.randint(0, commit.num_functions - 1)
        state_tensor[0][func_to_alter:, :, FUNCTION_ALTERED_LOC] = 1


class BugBuddyEnvironment(Env):
    '''
    An environment is the commit and interacting with that commit such as
    running tests against it and assigning blame
    '''
    def __init__(self, repository: Repository, synthetic_training=False):
        ''''
        Creates an environment from a commit
        '''
        # self.observation_space = spaces.Discrete(None)
        # self.action_space = spaces.Discrete(None)
        self.repository = repository
        self.synthetic_training = synthetic_training

        self.session = Session.object_session(repository)

    def set_commit(self, commit):
        '''
        Whenever we update the commit for the Environment, we need to set
        multiple different variables
        '''
        logger.info('Setting commit to {}'.format(commit))

        self.session = Session.object_session(commit)
        self.commit = commit
        self.state = commit_to_state(
            commit,
            synthetic_training=self.synthetic_training)

        print('State shape: ', self.state.shape)
        # All tests are linked with a corresponding TestRun instance.  When
        # running the tests against this state
        if not self.synthetic_training:
            self.test_run = create(self.session, TestRun, commit=self.commit)

        # list of available tests, sorted in order of their id.  It is sorted
        # by the Test.id
        self.all_tests = get_list_of_tests(self.commit)
        self.all_tests.sort(key=lambda test: test.id, reverse=False)

        self.total_rewards = 0

        # list of the tests that have already been ran for this commit
        self.tests_ran = []
        logger.info('Set commit')

    def step(self, action):
        '''
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (new_state, reward, done, info).

        # Arguments
            action (object): Agent chosen by the agent.  For example, the action
                             could be to run a test to see if it has newly
                             failed or newly passed status

        # Returns
            new_state (object): the new state of the current environment after
                                the action was taken
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further
                            step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for
                         debugging, and sometimes learning).
        '''
        logger.info('Running test: {}'.format(action))
        # for right now, the action will be the test that the agent wants to run

        # if we are in synthetic training, then we don't need to run the test.
        # We can simply retreive from the database whether or not the test is
        # failing.
        if self.synthetic_training:
            return self.synthetic_training_step(action)
        else:
            raise NotImplementedError(
                'Currently you need to set synthetic_training to True since'
                'I have not implemented non-synthetic training steps yet')

    def get_reward(self, test_result: TestResult):
        '''
        Returns the reward for the action.  The goal is to reward the agent so
        that it runs in the tests in the following order:

            1) Run all newly passing or failing tests
            2) Run the rest of the tests

        The rewards are calculated in the
        following way:

            1) Discover newly failing or newly passing test == +1 - percentage
               of tests already ran that were not newly passing/failing.  This
               encourages the newly failing or newly passing tests to be ran
               first.

               For example, if the 6th test ran was newly failing, but 2 out of
               the 5 tests that were already ran had no change in test status,
               then the reward would be:

                  r = 1 - num_not_newly_failing_or_passing / num_tests_ran
                  r = 1 - (2 / 5)
                  r = 3/5 or 0.6

               For example, if the 301st test ran was newly failing, but 290 out
               of the 300 test that were already ran had no change in test
               status, then the reward would be:

                  r = 1 - num_not_newly_failing_or_passing / num_tests_ran
                  r = 1 - (290 / 300)
                  r = 1 - 0.96
                  r = 0.04

            2) No change in test status == 0

            3) Ran a test that was already ran in the test run == -10
        '''
        # if the commit has already ran the test, then we should heavily
        # punish the agent for being wasteful.  What is currently stored in the
        # state for the test should ideally be TEST_OUTPUT_NOT_RUN.
        if (self.get_current_status(test_result.test) != TEST_OUTPUT_NOT_RUN):
            reward = -10

        # If the status is different from the previous commit's status and we
        # have already ruled out skips, then we should positively reward the
        # agent while also subtracting how many non-newly failing/passing tests
        # have already been ran.
        elif (self.test_status_changed(test_result.test)):
            # If this is out first test ran and it has a changed status
            if not self.tests_ran:
                reward = 1

            # get the number of tests that have already been ran that had a new
            # status in comparison with the previous commit
            num_newly_changed_status = sum(
                1 for test in self.tests_ran
                if self.test_status_changed(test))

            reward = 1 - (num_newly_changed_status / len(self.tests_ran))

        # If the status is the same in this commit as it was in the previous
        # commit, simply return 0.
        else:
            reward = 0

        self.total_rewards += reward
        return reward

    def test_status_changed(self, test: Test) -> str:
        '''
        Returns if the status for a test changed from the previous commit to the
        current commit
        '''
        return self.get_current_status(test) == self.get_previous_status(test)

    def get_current_status(self, test: Test) -> str:
        '''
        Returns the current status in the state vector for a test
        '''
        # the location of the test in the state tensor
        state_test_id = self.get_state_test_id(test)
        status_id = int(self.state[0][0][state_test_id][TEST_STATUS_LOC])
        status_lookup = {value: key for key, value in
                         TEST_STATUS_TO_ID_MAP.items()}
        return status_lookup[status_id]

    def get_previous_status(self, test: Test) -> str:
        '''
        Returns the previous status in the state vector for a test
        '''
        # the location of the test in the state tensor
        state_test_id = self.get_state_test_id(test)
        status_id = int(self.state[1][0][state_test_id][TEST_STATUS_LOC])
        status_lookup = {value: key for key, value in
                         TEST_STATUS_TO_ID_MAP.items()}
        return status_lookup[status_id]

    def update_state(self, test_result: TestResult):
        '''
        Given the output of a test, update the state
        '''
        logger.info('Updating state for {test} with status {status}'
                    .format(test=test_result.test,
                            status=test_result.status))
        result_id = TEST_STATUS_TO_ID_MAP[test_result.status]
        test_index = self.get_state_test_id(test_result.test)
        self.state[0][:, test_index:test_index, TEST_STATUS_LOC] = result_id

    def get_state_test_id(self, test):
        '''
        Given a test, it will return the state id for that test.  The state id
        represents the corresponding location in the state tensor for the test.
        It is ordered by the Test.id.
        '''
        try:
            return self.all_tests.index(test)
        except ValueError:
            import pdb; pdb.set_trace()

    @property
    def done(self):
        '''
        Whether or not this episode is complete.  An episode is complete for a
        given commit if all the tests are ran against the commit.  Note that
        a test can be ran multiple times by the agent, so we should not count
        duplicates.
        '''
        return len(list(set(self.tests_ran))) == len(self.all_tests)

    def get_synthetic_test_result(self, test_to_run):
        '''
        Given a test to run, look and see for the synthetic test's history what
        the result is
        '''
        assert len(self.commit.test_runs) == 1
        test_run = self.commit.test_runs[0]
        for test_result in test_run.test_results:
            if test_result.test.id == test_to_run.id:
                return test_result

    def synthetic_training_step(self, action_id):
        '''
        Returns a tuple containing (new_state, reward, done, info)

        @param: action_id.  This is the index of the test in the list of sorted
        tests for a commit.
        '''
        test = self.all_tests[action_id]

        test_result = self.get_synthetic_test_result(test)
        assert test_result

        # determine the reward for running that particular test
        reward = self.get_reward(test_result)

        # update the state with the action
        self.update_state(test_result)

        # we can store extra debugging information here
        info_dict = {}
        logger.info('{commit} -- {test} -> {status} | previously: {previous_status} | '
                    'reward: {reward} | total: {total}'
                    .format(commit=self.commit,
                            test=test,
                            previous_status=self.get_previous_status(test),
                            status=test_result.status,
                            reward=reward,
                            total=self.total_rewards))
        return self.state, reward, self.done, info_dict

    def reset(self):
        '''
        Resets the state of the environment and returns an initial state.

        # Returns
            state (object): The initial state of the space. Initial reward is
                            assumed to be 0.
        '''
        if self.synthetic_training:
            # retrieve a random synthetic commit and set the environment to
            # that commit
            commit = (
                self.session.query(Commit)
                .filter(Commit.commit_type == SYNTHETIC_CHANGE)
                .filter(Commit.repository_id == self.repository.id)
                .order_by(func.random())
                .first())

            # re-initialize the environment with the new commit
            self.set_commit(commit)

            return self.state
        else:
            raise NotImplementedError(
                'In _reset you need to set synthetic_training to True since'
                'non-synthetic training steps are not implemented')

    def render(self, mode='human', close=False):
        '''
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        '''
        raise NotImplementedError()


class BugBuddyProcessor(Processor):
    '''
    acts as a coupling mechanism between an Brain's agent and the environment
    '''

    def process_state_batch(self, batch):
        '''
        Given a state batch, I want to remove the second dimension, because it's
        useless and prevents me from feeding the tensor into my CNN
        '''
        return numpy.squeeze(batch, axis=1)


class Brain(object):
    '''
    Class that makes decisions on which tests should be ran and which source
    code should be blamed
    '''
    def __init__(self, repository: Repository):
        '''
        Initialize the Brain
        '''
        self.weights_filename = 'brain_weights.h5f'

        memory = SequentialMemory(limit=1000, window_length=1)
        processor = BugBuddyProcessor()

        number_of_tests = len(repository.tests)

        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            value_max=1.,
            value_min=.1,
            value_test=.05,
            nb_steps=100000)

        # Create the agent
        model = Sequential()
        model.add(Convolution3D(128,
                                (4, 4, 4),
                                strides=(3, 3, 3),
                                # input_shape=get_input_shape(repository)))
                                input_shape=(5, 337, 513, 3)))
        model.add(Activation('relu'))
        # model.add(Convolution3D(128, (3, 3, 3), strides=(2, 2, 2)))
        # model.add(Activation('relu'))
        model.add(Convolution3D(32, (1, 1, 1), strides=(1, 1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(number_of_tests))
        model.add(Activation('linear'))

        self.agent = DQNAgent(
            model=model,
            nb_actions=number_of_tests,
            policy=policy,
            memory=memory,
            processor=processor,
            nb_steps_warmup=5000,
            gamma=.99,
            target_model_update=1000,
            train_interval=4,
            delta_clip=1.)

        self.agent.compile(
            Adam(lr=.00025),
            metrics=['mae'])

        print(model.summary())

    def train(self, env):
        '''
        Trains the agent
        '''
        checkpoint_weights_filename = 'brain_weights_{step}.h5f'
        log_filename = 'brain_log.json'
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename,
                                             interval=25000)]
        callbacks += [FileLogger(log_filename, interval=100)]

        # Training our agent
        self.agent.fit(env,
                       callbacks=callbacks,
                       nb_steps=175000,
                       log_interval=10000)

        # After training is done, we save the final weights one more time.
        self.agent.save_weights(self.weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        self.agent.test(env, nb_episodes=10, visualize=False)

    def run(self, env):
        '''
        Performs actions on a commit, such as running tests
        '''
        self.agent.load_weights(self.weights_filename)
        self.agent.test(env, nb_episodes=10, visualize=True)
