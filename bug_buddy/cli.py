'''
Utilities for communicating with the user via the command line output
'''
from blessings import Terminal


def is_affirmative(user_response: str):
    '''
    Utility for determining if the user responded positively or not
    '''
    return user_response == 'y' or user_response == 'yes'
