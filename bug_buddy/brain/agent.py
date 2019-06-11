'''
The agent that interacts with the environment
'''
from __future__ import division
import numpy
from rl.agents.dqn import DQNAgent
from rl.core import Processor


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


class BugBuddyAgent(DQNAgent):
    '''
    Wrapper around the Keras RL Agent
    '''
