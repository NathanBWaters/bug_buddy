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
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from bug_buddy.constants import (
    TEST_OUTPUT_SUCCESS,
    TEST_OUTPUT_FAILURE,
    TEST_OUTPUT_SKIPPED)
from bug_buddy.schema import Repository, Commit


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


def get_previous_commits(commit, num_commits=4):
    '''
    Gets the commit's previous commits
    '''
    raise NotImplementedError()


def commit_to_feature(commit: Commit):
    '''
    Converts a commit to a numpy array that represents the features for
    determining which test failures might occur from the change.

    In this case, it is simply a numpy array with a 1 for each method that was
    changed and a 0 for each method that was not changed.  The array is in
    alphabetical order.
    '''
    # TODO - this doesn't make sense when a commit could have different
    # functions (i.e. adding functions, etc)
    sorted_functions = commit.repository.functions
    sorted_functions.sort(key=lambda func: func.id, reverse=False)

    feature = []
    for function in sorted_functions:
        # if the function even has a diff for the commit, then it's the
        # 'assert False' statement
        was_altered = any([diff.commit.id == commit.id for diff in
                           function.diffs])
        feature.append(int(was_altered))

    if len(feature) != 337:
        import pdb; pdb.set_trace()

    test_results = commit.test_runs[0].test_results
    sorted_tests.sort(key=lambda test: test.id, reverse=False)

    # for test in sorted_tests:
        # if test.
    return numpy.asarray(feature)


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
