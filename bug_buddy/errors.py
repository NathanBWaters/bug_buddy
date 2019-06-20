'''
Contains the bug_buddy errors
'''


class BugBuddyError(Exception):
    '''
    Base class for all bug_buddy errors
    '''


class UserError(BugBuddyError):
    '''
    Class for raising user errors
    '''
