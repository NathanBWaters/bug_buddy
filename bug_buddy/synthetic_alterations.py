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
    powerset,
    synthetic_blame,
    get_hash_given_base_synthetic_ids)
from bug_buddy.constants import (
    BASE_SYNTHETIC_CHANGE,
    BENIGN_STATEMENT,
    ERROR_STATEMENT,
    PYTHON_FILE_TYPE,
    SYNTHETIC_CHANGE,
    SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import session_manager, Session, create, get
from bug_buddy.errors import UserError
from bug_buddy.git_utils import (
    create_commit,
    create_reset_commit,
    db_and_git_match,
    get_previous_commit,
    git_push,
    is_repo_clean,
    revert_unstaged_changes,
    revert_to_master,
    run_cmd,
    set_bug_buddy_branch,
    update_commit)
from bug_buddy.runner import run_all_tests, library_is_testable
from bug_buddy.logger import logger
from bug_buddy.schema import (
    Commit,
    Diff,
    Function,
    FunctionHistory,
    Repository,
    TestRun)
from bug_buddy.snapshot import snapshot_commit, create_diffs, save_diffs
from bug_buddy.source import (
    apply_diff,
    get_function_nodes_from_repo,
    revert_diff)


def yield_blame_set(synthetic_diffs: DiffList):
    '''
    Returns a set of diffs.

    1) Returns each synthetic diff of a repository one by one
    2) Once all diffs have been returned individually, it will then returns a
       set of 4 diffs that were randomly chosen.
    '''
    while True:
        diff_set = []
        for i in range(4):
            diff_set.append(
                synthetic_diffs[random.randint(0, len(synthetic_diffs) - 1)])

        logger.info('Yielding diff set: {}'.format(diff_set))
        # remove duplicates if they exist
        yield list(set(diff_set))


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    session = Session.object_session(repository)
    synthetic_diffs = repository.get_synthetic_diffs()

    if not synthetic_diffs:
        # create the synthetic diffs
        create_synthetic_alterations(repository)
        logger.info('You have created the base synthetic commits.  Congrats!')
        session.commit()
        synthetic_diffs = repository.get_synthetic_diffs()

    num_runs = 0
    for diff_set in yield_blame_set(synthetic_diffs):
        logger.info('On diff set: {}'.format(diff_set))

        for diff_subset in powerset(diff_set):
            logger.info('On run #{} with: {}'.format(num_runs, diff_subset))
            db_and_git_match(repository)

            try:
                # see if we already have a commit and test run for the diff set.
                # if we do, continue
                logger.debug('1: {}'.format(time.time()))
                commit = get_matching_commit_for_diffs(repository, diff_subset)

                # if the commit does not already exist for this set, then we
                # need to create it and run tests against it
                if not commit:
                    # revert back to a clean repository
                    reset_commit = create_reset_commit(repository)
                    if reset_commit:
                        logger.info('Storing reset commit')
                        snapshot_commit(repository, reset_commit)
                        git_push(repository)
                        session.commit()
                        db_and_git_match(repository)

                    # create a commit.  Only allow an empty commit if there
                    # nothing in the diff
                    commit = create_commit(repository,
                                           name=SYNTHETIC_CHANGE,
                                           commit_type=SYNTHETIC_CHANGE,
                                           allow_empty=True)

                    logger.info('Applying diffs')
                    # apply the synthetic diffs to the mirrored repository
                    apply_synthetic_diffs(commit, diff_subset)

                    # store the rest of the commit data.  No need to recreate
                    # the diffs since they have already been stored in
                    # apply_synthetic_diffs
                    logger.info('Snapshotting the commit: {}'.format(commit))
                    commit = snapshot_commit(
                        repository,
                        commit,
                        skip_diffs=True)

                # add the commit hash id for its synthetic diffs
                logger.info('Creating synthetic diff hash')
                if not commit.synthetic_diff_hash:
                    base_synthetic_ids = [diff.id for diff in diff_subset]
                    commit.synthetic_diff_hash = (
                        get_hash_given_base_synthetic_ids(base_synthetic_ids))
                    logger.info('Added hash_ids #{} to commit: {}'
                                .format(commit.synthetic_diff_hash, commit))

                logger.info('Running tests')
                if not commit.test_runs:
                    # run all tests against the synthetic change
                    run_all_tests(commit)

                logger.debug('2: {}'.format(time.time()))
                if commit.needs_blaming():
                    synthetic_blame(commit, commit.test_runs[0])

                logger.debug('3: {}'.format(time.time()))
                session.commit()

                # push newly created commit
                git_push(repository)

                logger.info('Completed run #{}'.format(num_runs))

                num_runs += 1
                if run_limit and num_runs >= run_limit:
                    logger.info('Completed all #{} runs! '.format(num_runs))
                    exit()

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


def create_synthetic_alterations(repository: Repository):
    '''
    Creates synthetic changes to a code base, creates a commit, and then runs
    the tests to see how the changes impacted the test results.  These changes
    are either 'assert False' or 'assert True'.

    @param repository: the code base we are changing
    @param commit: the empty commit we're adding changes to
    '''
    session = Session.object_session(repository)
    # create an empty commit that the diffs will be added to
    commit = create_commit(
        repository,
        name='base_synthetic_change',
        commit_type=BASE_SYNTHETIC_CHANGE,
        allow_empty=True)

    function_nodes = get_function_nodes_from_repo(repository)
    for node in function_nodes:
        create_synthetic_diff_for_node(
            repository,
            commit,
            node)

    git_push(repository)

    # We want to checkpoint here in case it fails.  Greating synthetic
    # can take a while
    session.commit()


def get_matching_commit_for_diffs(repository, base_synthetic_diffs):
    '''
    Given a set of diffs, return if there is a commit that has those diffs
    '''
    session = Session.object_session(repository)
    base_synthetic_ids = [diff.id for diff in base_synthetic_diffs]
    diff_hash = get_hash_given_base_synthetic_ids(base_synthetic_ids)
    return get(session, Commit, synthetic_diff_hash=diff_hash)


def apply_synthetic_diffs(commit: Commit, diff_subset: DiffList):
    '''
    Creates a new diff from the base synthetic diff.  It then stores the newly
    created diff
    '''
    for base_synthetic_diff in diff_subset:
        # create Diff instances
        apply_diff(base_synthetic_diff)

        # save the diffs but only look for the unstaged edits
        new_diffs = create_diffs(commit.repository, commit, only_unstaged=True)

        try:
            # there should only be one created
            msg = ('More than one diff created in the apply_synthetic_diff '
                   'step. The diffs are: {}'.format(new_diffs))
            assert len(new_diffs) == 1, msg
        except AssertionError:
            print('Got more than one diff')
            import pdb; pdb.set_trace()
            create_diffs(commit.repository, commit, only_unstaged=True)

        new_diffs[0].base_synthetic_diff_id = base_synthetic_diff.id
        new_diffs[0].function = base_synthetic_diff.function
        save_diffs(commit.repository, commit, new_diffs)

        # Now add the update to change to the commit so it is not noticed
        # in the next diff creation.  This will also change the commit_id of the
        # commit
        update_commit(commit)


def create_synthetic_diff_for_node(repository: Repository,
                                   commit: Commit,
                                   node):
    '''
    Creates the visited function and adds an 'assert False' to the node.
    This is used for creating synthetic 'assert False' diffs for each function.
    '''
    session = Session.object_session(repository)

    previous_commit = get_previous_commit(commit)
    function = previous_commit.get_function_for_node(node).function

    # create the function history instance
    function_history = create(
        session,
        FunctionHistory,
        function=function,
        commit=commit,
        node=node,
        first_line=node.first_line,
        # we need the minus 1 because when we complete the commit the
        # 'assert False' line will have been removed
        last_line=node.last_line - 1,
    )

    logger.info('There is a new function history: {}'.format(function_history))

    added_line = function_history.prepend_statement('assert False')

    if library_is_testable(repository):
        # create a new diff from this one change
        diffs = create_diffs(
            repository,
            commit=commit,
            function=function,
            is_synthetic=True,
            allow_empty=False)

        # There should always be only one diff created from altering one
        # function
        assert len(diffs) == 1

        diff = diffs[0]
        logger.info('Created diff: {}'.format(diff))

        # go back to a clean repository
        revert_diff(diff)

    else:
        # remove the addition from the source code
        function_history.remove_line(added_line)

    return node
