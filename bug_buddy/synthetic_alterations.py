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
from bug_buddy.errors import BugBuddyError
from bug_buddy.execution import run_test
from bug_buddy.git_utils import (is_repo_clean,
                                 create_commit,
                                 revert_commit)
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Routine, TestRun


def generate_synthetic_test_results(repository: Repository,
                                    run_limit: int=1):
    '''
    Creates multiple synthetic changes and test results
    '''
    print('Creating synthetic results for: ', repository)
    num_runs = 0
    while run_limit is None or num_runs <= run_limit:
        print('Creating TestRun #{}'.format(num_runs))
        create_synthetic_change_and_fixing_changes(repository)
        num_runs += 1
        break


def create_synthetic_change_and_fixing_changes(repository: Repository):
    '''
    Creates synthetic changes to a code base, creates a commit, and then runs
    the tests to see how the changes impacted the test results.  These changes
    are either 'assert False' or 'assert True'.

    It then creates a series of 'Fixing Changes', which individual revert one of
    the 'assert False' statements.  For each revert of the 'assert False'
    statements, it will then rerun the tests to see how a fixing change alters
    the test results

    @param repository: the code base we are changing
    '''
    if not is_repo_clean(repository):
        msg = ('You attempted to work on an unclean repository.  Please run: \n'
               '"git checkout ." to clean the library')
        raise BugBuddyError(msg)
    edit_random_function(repository)
    commit = create_commit(repository)
    logger.info('Created commit: {}'.format(commit))
    # test_run = run_test(repository, commit)
    # revert_commit(repository)


def edit_random_function(repository):
    '''
    Alters the repository in a very simplistic manner.  For right now, we are
    just going to take a method or function and add either an assert False or
    assert True to it

    @param repository: the code base we are changing
    '''
    # contains the methods/functions across the files
    routines = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        routines.extend(get_routines_from_file(repository, repo_file))

    selected_routine = routines[random.randint(0, len(routines) - 1)]
    _add_assert_to_routine(selected_routine)


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
    is_benign_statement = random.randint(0, 1)
    statement = BENIGN_STATEMENT if is_benign_statement else ERROR_STATEMENT
    routine.prepend_statement(statement)
