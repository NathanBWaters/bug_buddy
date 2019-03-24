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
import time
from typing import List

from bug_buddy.schema.aliases import DiffList
from bug_buddy.blaming import (
    get_matching_commit_for_diffs,
    powerset,
    synthetic_blame,
    get_diff_set_hash)
from bug_buddy.constants import (BENIGN_STATEMENT,
                                 ERROR_STATEMENT,
                                 PYTHON_FILE_TYPE,
                                 SYNTHETIC_CHANGE,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import session_manager, Session
from bug_buddy.errors import UserError
from bug_buddy.git_utils import (
    create_commit,
    create_reset_commit,
    git_push,
    is_repo_clean,
    revert_unstaged_changes,
    revert_to_master,
    run_cmd,
    set_bug_buddy_branch,
    update_commit)
from bug_buddy.runner import run_test, library_is_testable
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Function, TestRun, Commit, Diff
from bug_buddy.snapshot import snapshot_commit
from bug_buddy.source import (
    apply_diff,
    create_diffs,
    create_synthetic_alterations)


def yield_blame_set(repository: Repository):
    '''
    Returns a set of diffs.

    1) Returns each synthetic diff of a repository one by one
    2) Once all diffs have been returned individually, it will then returns a
       set of 4 diffs that were randomly chosen.
    '''
    diffs = repository.diffs

    while True:
        diff_set = []
        for i in range(4):
            diff_set.append(diffs[random.randint(0, len(diffs))])

        logger.info('Yielding diff set: {}'.format(diff_set))
        # remove duplicates if they exist
        yield list(set(diff_set))


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    session = Session.object_session(repository)

    num_runs = 0
    for diff_set in yield_blame_set(repository):
        logger.info('On diff set: {}'.format(diff_set))

        for diff_subset in powerset(diff_set):
            logger.info('On run #{} with: {}'.format(num_runs, diff_subset))

            try:
                # see if we already have a commit and test run for the diff set.
                # if we do, continue
                logger.debug('1: {}'.format(time.time()))
                commit = get_matching_commit_for_diffs(repository, diff_subset)

                # if the commit does not already exist for this set, then we
                # need to create it and run tests against it
                if not commit:
                    logger.info('Creating a new commit for diff_subset: {}'
                                .format(diff_subset))

                    # revert back to a clean repository
                    create_reset_commit(repository)

                    # create a commit.  Only allow an empty commit if there nothing
                    # in the diff
                    commit = create_commit(repository,
                                           commit_type=SYNTHETIC_CHANGE,
                                           allow_empty=True)
                    # apply the synthetic diffs
                    for diff in diff_subset:
                        apply_diff(diff)

                    # save the diffs with the commit
                    # TODO - need to recreate the diffs but now they're tied to
                    # the correct commit
                    raise Exception('Yeah you need to save the diffs against '
                                    'the commit')

                # add the commit hash id for its synthetic diffs
                if not commit.synthetic_diff_hash:
                    commit.synthetic_diff_hash = get_diff_set_hash(diff_subset)
                    logger.info('Added hash_ids #{} to commit: {}'
                                .format(commit.synthetic_diff_hash, commit))

                if not commit.test_runs:
                    # run all tests against the synthetic change
                    run_test(commit)

                logger.debug('2: {}'.format(time.time()))
                if commit.needs_blaming():
                    synthetic_blame(commit, commit.test_runs[0])

                logger.debug('3: {}'.format(time.time()))
                session.commit()

                # push newly created commit
                # git_push(repository)

                logger.info('Completed run #{}'.format(num_runs))

                num_runs += 1
                if run_limit and num_runs >= run_limit:
                    logger.info('Completed all #{} runs.  Exiting'.format(num_runs))
                    break
            except Exception as e:
                # revert all the local edits
                logger.error('Hit the following exception: {}'.format(e))
                logger.error('Reverting local changes')
                revert_to_master(repository)
                raise e


def _get_assert_statement(repo_function):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    is_error_statement = random.randint(0, 1)
    return ERROR_STATEMENT if is_error_statement else BENIGN_STATEMENT
