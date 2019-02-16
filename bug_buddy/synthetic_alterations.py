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
from bug_buddy.blaming import blame
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
from bug_buddy.snapshot import snapshot_commit, snapshot_test_results
from bug_buddy.source import create_synthetic_alterations


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    with session_manager() as session:
        session.add(repository)

        # Adds a random number of edits to the repository.
        if not repository.diffs:
            # adds 'assert False' to each function
            logger.info('Creating synthetic alterations')
            create_synthetic_alterations(repository)

            # creates a diff for each 'assert False'
            logger.info('Creating diffs')
            create_diffs(repository)

        num_runs = 0
        for diff_set in powerset(repository.diffs):
            logger.info('On DiffSet #{} with: {}'.format(num_runs, diff_set))

            # revert back to a clean repository
            reset_branch(repository)

            # apply diffs
            for diff in diff_set:
                add_diff(diff)

            commit = create_commit(repository)

            # Store the data of the commit in the database.  Create the
            # DiffCommitLink instances
            snapshot_commit(repository, commit)

            # run all tests against the synthetic change
            test_run = run_test(repository, commit)

            # store the relationship between the test results and the functions
            snapshot_test_results(repository, commit, test_run)

            # determine which diffs caused which test failures
            # blame(repository, commit, test_run)

            # push all the new commits we've created
            git_push(repository)

            num_runs += 1


def create_fixing_change(repository: Repository, diffs: DiffList):
    '''
    Creates a single fixing change.  It removes a single 'assert False' that
    is still present from the original synthetic alteration
    '''
    random.shuffle(diffs)

    made_fixing_alteration = False
    for diff in diffs:
        if ERROR_STATEMENT in diff.content:
            logger.info('Creating fixing change by removing line {} from {}'
                        .format(diff.line_number, diff.file))
            # we have an error statement that we need to remove.
            file_path = os.path.join(repository.path, diff.file)
            with open(file_path) as f:
                content = f.readlines()

            content.pop(diff.line_number)

            with open(file_path, 'w') as f:
                f.writelines(content)

            made_fixing_alteration = True
            break

    # assumes we made a fix
    if made_fixing_alteration:
        commit = create_commit(repository,
                               name='synthetic_fixing_change',
                               commit_type=SYNTHETIC_FIXING_CHANGE)
        test_run = run_test(repository, commit)

        session = Session.object_session(repository)
        # add the commit and test run to our database
        session.add(commit)
        session.add(test_run)


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
