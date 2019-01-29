'''
Assigns blame for test failures to a commit
'''
from junitparser import JUnitXml

from bug_buddy.constants import SUCCESS, FAILURE
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit


def blame(repository: Repository, test_run: TestRun):
    '''
    Given a TestRun, it will determine which lines are to blame for each
    test failures.
    '''
    
