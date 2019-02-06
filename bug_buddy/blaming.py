'''
Assigns blame for test failures to a commit
'''
import itertools
from junitparser import JUnitXml

from bug_buddy.constants import ERROR_STATEMENT
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit


def blame(repository: Repository, commit: Commit, test_run: TestRun):
    '''
    Given a TestRun, it will determine which lines are to blame for each
    test failures.
    '''
    errored_diffs = [diff for diff in commit.diffs
                     if ERROR_STATEMENT in diff.content]
    for errored_diff_set in powerset(errored_diffs):
        if len(errored_diff_set) == 0:
            continue


def powerset(diffs):
    '''
    Returns the powerset of the diffs except the empty set

    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

    @param: list of diffs
    @returns: powerset of the diffs
    '''
    return (itertools.chain.from_iterable(
        itertools.combinations(diffs, index) for index in range(len(diffs) + 1)
    ))
