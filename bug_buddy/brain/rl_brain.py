'''
Reinforcement Learning brain
'''
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution3D
from keras.optimizers import Adam
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from bug_buddy.buddy.agent import BugBuddyAgent, BugBuddyProcessor
from bug_buddy.buddy.env import BugBuddyAgent, BugBuddyProcessor
from bug_buddy.schema import Repository
from bug_buddy.logger import logger


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

        memory = SequentialMemory(limit=1, window_length=1)
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

        self.agent = BugBuddyAgent(
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
        callbacks = [
            ModelIntervalCheckpoint(checkpoint_weights_filename, interval=25000),
            FileLogger(log_filename, interval=100)
        ]

        # Training our agent
        self.agent.fit(env,
                       callbacks=callbacks,
                       nb_steps=175000,
                       log_interval=10000,
                       verbose=0)  # just to remove the progress bar

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


def synthetic_train(repository: Repository):
    '''
    Trains the agent on synthetic data generation
    '''
    logger.info('Training on synthetic data for: {}'.format(repository))
    # logger.info('Caching commit tensors')
    # cache_commits(repository)

    logger.info('Initializing environment')
    brain = Brain(repository)
    env = ChangeEnvironment(repository, synthetic_training=True)
    env.reset()
    brain.train(env)
