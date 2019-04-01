'''
Utilities for communicating with the user via the command line output
'''
from blessings import Terminal

t = Terminal()

print(t.bold('Hi there!'))
print(t.bold_red_on_bright_green('It hurts my eyes!'))

with t.location(0, t.height - 1):
    print('This is at the bottom.')


def is_affirmative(user_response: str):
    '''
    Utility for determining if the user responded positively or not
    '''
    return user_response == 'y' or user_response == 'yes'
