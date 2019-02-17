#!/usr/bin/env python3
'''
API for editing a repository and generating synthetic data.

It follows the following idea for generating synthetic data:

Create baseline edits
 - For each chunk of code (method, function, etc), make either a benign or
   test-breaking edit
 - Run the tests, record which tests fail

Create composition edits:
 - Combine benign and test-breaking edits.  Since we already know how they
   each change affects the test, we know the "bad" line already.  Store this
   data.

This data will be trained upon where we will provide the changes and we already
know which line is really at fault for a failing test.
'''
import ast
import inspect
import importlib
import itertools
import os
import random
import re
import sys
from typing import List

from bug_buddy.schema.aliases import DiffList
from bug_buddy.blaming import synthetic_blame
from bug_buddy.constants import (BENIGN_STATEMENT,
                                 ERROR_STATEMENT,
                                 PYTHON_FILE_TYPE,
                                 SYNTHETIC_CHANGE,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import session_manager, Session
from bug_buddy.errors import UserError
from bug_buddy.git_utils import (add_diff,
                                 create_commit,
                                 create_reset_commit,
                                 create_diffs,
                                 git_push,
                                 is_repo_clean,
                                 revert_unstaged_changes,
                                 run_cmd,
                                 set_bug_buddy_branch,
                                 reset_branch,
                                 update_commit)
from bug_buddy.runner import run_test, library_is_testable
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Function, TestRun, Commit, Diff
from bug_buddy.snapshot import (snapshot_commit,
                                snapshot_diff_commit_link,
                                snapshot_initialization)
from bug_buddy.source import create_synthetic_alterations


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    with session_manager() as session:
        session.add(repository)

        snapshot_initialization(repository)

        num_runs = 0
        for diff_set in powerset(repository.diffs):
            logger.info('On DiffSet #{} with: {}'.format(num_runs, diff_set))

            # revert back to a clean repository
            reset_branch(repository)

            # apply diffs
            for diff in diff_set:
                add_diff(diff)

            # create a commit.  Only allow an empty commit if there nothing
            # in the diff
            commit = create_commit(repository, allow_empty=not diff_set)

            snapshot_diff_commit_link(commit, diff_set)

            # run all tests against the synthetic change
            run_test(repository, commit)

            # determine which diffs caused which test failures
            # synthetic_blame(repository, commit, test_run)

            # push all the new commits we've created
            git_push(repository)

            num_runs += 1


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


def _get_assert_statement(repo_function):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    is_error_statement = random.randint(0, 1)
    return ERROR_STATEMENT if is_error_statement else BENIGN_STATEMENT
