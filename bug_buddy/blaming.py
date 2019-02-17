'''
Assigns blame for test failures to a commit
'''
import itertools
from junitparser import JUnitXml

from bug_buddy.constants import (ERROR_STATEMENT,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.git_utils import create_commit
from bug_buddy.logger import logger
from bug_buddy.runner import run_test
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit
from bug_buddy.snapshot import snapshot_commit


def synthetic_blame(repository: Repository,
                    commit: Commit):
    '''
    Given a synthetic commit, it will create blames for the commit based on
    the blames of the sub-combinations of the diffs
    '''
    pass
