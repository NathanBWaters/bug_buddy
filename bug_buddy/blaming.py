'''
Assigns blame for test failures to a commit
'''
import itertools
from junitparser import JUnitXml

from bug_buddy.constants import (ERROR_STATEMENT,
                                 FAILURE,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.git_utils import create_commit
from bug_buddy.logger import logger
from bug_buddy.runner import run_test
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit
from bug_buddy.snapshot import snapshot_commit


def blame(repository: Repository,
          parent_commit: Commit,
          is_synthetic: bool=False):
    '''
    Given a commit, it will remove subsets of diffs of the commit until it has
    determined which subsets of commits need to be removed in order for each
    failing test to pass.  It uses the following logic:
        - Starting with the smallest subset of diffs
        - Revert all diffs in the subset of diffs
        - Create a commit
        - Snapshot that commit
        - Run all tests
        - If any test that has been failing so far is now passing, then that
          subset of diffs is to blame for the the failing.
        - revert back to the original blame, but do not creat a new commit.

    '''
    errored_diffs = [diff for diff in parent_commit.diffs
                     if ERROR_STATEMENT in diff.content]

    latest_test_run = parent_commit.test_runs[0]
    failing_tests = [test for test in latest_test_run.test_results
                     if test.status == FAILURE]
    for errored_diff_set in powerset(errored_diffs):
        if len(errored_diff_set) == 0:
            continue

        if not failing_tests:
            logger.info('All tests are now passing, you have succesfully '
                        'set blame for all the failing tests')
            return

        # revert each part of the diff in the set
        for diff in errored_diff_set:
            diff.revert()

        # create a new commit
        diff_commit = create_commit(
            repository,
            name='diff_subset_of_{}'.format(parent_commit.commit_id),
            commit_type=SYNTHETIC_FIXING_CHANGE)
        # Store the data of the commit in the database
        snapshot_commit(repository, diff_commit)

        # run the tests with the newly created commit
        test_run = run_test(repository, diff_commit)

        # get a list of failing tests.
        failing_tests = [test for test in test_run.test_results
                         if test.status == FAILURE]


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
