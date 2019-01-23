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
from importlib.machinery import SourceFileLoader
import os
import random
import re
import sys

from bug_buddy.constants import (BENIGN_STATEMENT,
                                 ERROR_STATEMENT,
                                 PYTHON_FILE_TYPE)
from bug_buddy.db import session_manager, Session
from bug_buddy.errors import BugBuddyError
from bug_buddy.git_utils import (is_repo_clean,
                                 create_commit,
                                 revert_commit,
                                 set_bug_buddy_branch)
from bug_buddy.runner import run_test
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Routine, TestRun


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    logger.info('Creating synthetic results for: {}'.format(repository))
    num_runs = 0
    while run_limit is None or num_runs <= run_limit:
        logger.info('Creating TestRun #{}'.format(num_runs))

        create_synthetic_alterations_and_change(repository)
        # create_fixing_changes(repository)
        num_runs += 1
        break


def create_fixing_changes(repository: Repository):
    '''
    Creates fixing changes for the synthetic change
    '''
    print('Implement run_test')
    assert False


def create_synthetic_alterations_and_change(repository: Repository):
    '''
    Creates synthetic changes to a code base, creates a commit, and then runs
    the tests to see how the changes impacted the test results.  These changes
    are either 'assert False' or 'assert True'.

    @param repository: the code base we are changing
    '''
    if not is_repo_clean(repository):
        msg = ('You attempted to work on an unclean repository.  Please run: \n'
               '"git checkout ." to clean the library')
        raise BugBuddyError(msg)

    # make sure we are on the bug buddy branch
    set_bug_buddy_branch(repository)

    # make synthetic alterations to the project
    edit_random_routines(repository)

    session = Session.object_session(repository)

    commit = create_commit(repository,
                           name='synthetic_alteration_change',
                           is_synthetic=True)

    test_run = run_test(repository, commit)

    # add the commit and test run to our database
    session.add(commit)
    session.add(test_run)


def get_routines_from_repo(repository):
    '''
    Returns the routines from the repository src files
    '''
    routines = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        routines.extend(get_routines_from_file(repository, repo_file))

    return routines


def edit_random_routines(repository, num_edits=None):
    '''
    Alters the repository in a very simplistic manner.  For right now, we are
    just going to take a method or function and add either an assert False or
    assert True to it

    @param repository: the code base we are changing
    '''
    # contains the methods/functions across the files
    # import pdb; pdb.set_trace()
    uneditted_routines = get_routines_from_repo(repository)

    altered_routines = []

    if num_edits is None:
        # at most edit only 1/10th of the routines in the repository
        num_edits = random.randint(1, int(len(repository.get_src_files()) / 10))

    for i in range(num_edits):
        routine_index = random.randint(0, len(uneditted_routines) - 1)
        selected_routine = uneditted_routines[routine_index]
        _add_assert_to_routine(selected_routine)

        altered_routines.append(selected_routine)

        # the file has been editted.  This means we need to refresh the routines
        # with the correct line numbers.  However, we still don't want to edit
        # the routine that we just previously altered.
        uneditted_routines = get_routines_from_repo(repository)

        for altered_routine in altered_routines:
            matching_routines = [
                routine for routine in uneditted_routines
                if routine.node.name == altered_routine.node.name]

            closest_routine = matching_routines[0]
            for matching_routine in matching_routines:
                if (abs(matching_routine.node.lineno - altered_routine.node.lineno) <
                        abs(closest_routine.node.lineno - altered_routine.node.lineno)):
                    # we have found a routine that is more likely to correspond
                    # with the original altered_routine.
                    closest_routine = matching_routine

            # delete the already altered routine from the list of available
            # routines
            uneditted_routines.remove(closest_routine)


def get_routines_from_file(repository, repo_file):
    '''
    Returns the methods and functions from the file
    '''
    routines = []

    with open(repo_file) as file:
        repo_file_content = file.read()
        repo_module = ast.parse(repo_file_content)
        for node in ast.walk(repo_module):
            if isinstance(node, ast.FunctionDef):
                routine = Routine(node, repo_file)
                routines.append(routine)

    return routines


def _add_assert_to_routine(routine):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    is_error_statement = random.randint(0, 1)
    statement = ERROR_STATEMENT if is_error_statement else BENIGN_STATEMENT
    routine.prepend_statement(statement)
