'''
This is the core class for performing operations on a code repository such as:
    1) Running a test
    2) Assigning blame
    3) Binary searching through a code repository to uncover bugs

'''
from __future__ import division
from PIL import Image
import numpy as np
import gym
from gym import Env, spaces
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K
import numpy
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from sqlalchemy import desc

from bug_buddy.constants import (
    SYNTHETIC_CHANGE,
    TEST_OUTPUT_SUCCESS,
    TEST_OUTPUT_FAILURE,
    TEST_OUTPUT_NOT_RUN,
    TEST_OUTPUT_SKIPPED)
from bug_buddy.db import Session
from bug_buddy.schema import Repository, Commit, Blame


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def get_blame_count(function, test):
    '''
    Returns the number of times a function change has been blamed for a test
    '''
    # A blame is made up of diff id and test result id, so given a test and a
    # function means we have to do some expensive queries
    session = Session.object_session(function)
    return len(session.query(Blame)
                      .filter(Blame.function_id == function.id)
                      .filter(Blame.test_id == test.id)
                      .all())


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
    '''
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

    test_status_map = {
        TEST_OUTPUT_NOT_RUN: 0,
        TEST_OUTPUT_SKIPPED: 1,
        TEST_OUTPUT_FAILURE: 2,
        TEST_OUTPUT_SUCCESS: 3,
    }

    for i in range(len(sorted_functions)):
        function = sorted_functions[i]
        function_was_altered = any([diff.commit.id == commit.id for diff in
                                    function.diffs])
        for j in range(len(sorted_tests)):
            # Step 1 - add whether or not the function was altered for this
            # commit.  1 for altered, 0 otherwise.
            commit_tensor[i][j][0] = int(function_was_altered)

            # Step 2 - add the status of the test.  If the test is not ran
            # the id will be 0, which represents that the test has not been
            # ran yet
            test = sorted_tests[i]
            test_results = [test_result for test_result in
                            commit.test_runs[0].test_results
                            if test_result.test.id == test.id]

            if len(test_results) > 1:
                import pdb; pdb.set_trace()

            if test_results:
                test_result = test_results[0]
                commit_tensor[i][j][1] = test_status_map[test_result.status]

            # Step 3 - add the blame count, which represents how many times
            # the function has been blamed for the test
            blame_count = get_blame_count(test, function)
            commit_tensor[i][j][2] = blame_count

    return commit_tensor


def get_previous_commits(commit: Commit):
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


def commit_to_input(commit: Commit):
    '''
    Converts a commit to a tensor that stores features about the commit and
    its previous commits
    '''
    commits = get_previous_commits(commit)
    commits.append(commit)

    input_tensor = []
    for commit in commits:
        input_tensor.append(commit_to_tensor(commit))

    numpy.asarray(input_tensor)


class BugBuddyEnvironment(Env):
    '''
    An environment is the commit and interacting with that commit such as
    running tests against it and assigning blame
    '''
    def __init__(self, commit, is_synthetic_training=False):
        ''''
        Creates an environment from a commit
        '''
        self.observation_space = spaces.Discrete(None)
        self.action_space = spaces.Discrete(None)

        self.is_synthetic_training = is_synthetic_training

    def step(self, action):
        '''
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (new_state, reward, done, info).

        # Arguments
            action (object): Agent chosen by the agent.  For example, the action
                             could be to run a test to see if it has newly
                             failed or newly passed

        # Returns
            new_state (object): the new state of the current environment after the
                                action waas taken
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further
                            step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for
                         debugging, and sometimes learning).
        '''
        # for right now, the action will be the test that the agent wants to run

        # if we are in synthetic training, then we don't need to run the test.
        # We can simply retreive from the database whether or not the test is
        # failing.
        if self.is_synthetic_training:
            return self.synthetic_training_step(action)
        else:
            raise NotImplementedError(
                'Currently you need to set is_synthetic_training to True since'
                'I have not implemented non-synthetic training steps yet')

    def get_reward(self, action):
        '''
        Returns the reward for the action
        '''
        pass

    def create_state(self, commit):
        '''
        The state is made up of the known test output of the last 4 commits as
        well as the current commit.
        The state is also the input into the Agent's neural network.
        '''
        commits = get_previous_commits(commit)
        commits.append(commit)

        state = []
        for index in range(commits):
            commit = commits[index]
            state[index] = commit.to_state_vector()

    def update_state(self, test_output):
        '''
        Given the output of a test, update the state
        '''

    def synthetic_training_step(self, action):
        '''
        Returns a tuple containing (new_state, reward, done, info)
        '''
        test_to_run = action

    def reset(self):
        '''
        Resets the state of the environment and returns an initial state.

        # Returns
            state (object): The initial state of the space. Initial reward is assumed to be 0.
        '''
        raise NotImplementedError()

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


class BugBuddyProcessor(object):
    '''
    acts as a coupling mechanism between an Brain's agent and the environment
    '''
    pass


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

        memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
        processor = BugBuddyProcessor()

        number_of_tests = 513

        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr='eps',
            value_max=1.,
            value_min=.1,
            value_test=.05,
            nb_steps=1000000)

        # Create the agent
        model = Sequential()
        model.add(Convolution2D(256, (8, 8), strides=(4, 4)))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(number_of_tests))
        model.add(Activation('linear'))
        print(model.summary())

        dqn = DQNAgent(
            model=self.model,
            nb_actions=1231,
            policy=policy,
            memory=memory,
            processor=processor,
            nb_steps_warmup=50000,
            gamma=.99,
            target_model_update=10000,
            train_interval=4,
            delta_clip=1.)
        dqn.compile(
            Adam(lr=.00025),
            metrics=['mae'])

    def train(self, env):
        '''
        Trains the agent
        '''
        checkpoint_weights_filename = 'brain_weights_{step}.h5f'
        log_filename = 'brain_log.json'
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]

        self.agent.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

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
