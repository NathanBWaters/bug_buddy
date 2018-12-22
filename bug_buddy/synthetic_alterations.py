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
import random
import re

from bug_buddy.constants import PYTHON_FILE_TYPE
from bug_buddy.errors import BugBuddyError
from bug_buddy.schema import Repository, TestRun
from bug_buddy.execution import run_test
from bug_buddy.git_utils import (is_repo_clean,
                                 create_commit,
                                 revert_commit)


def generate_synthetic_test_results(repository: Repository,
                                    run_limit: int=None):
    '''
    Creates multiple synthetic changes and test results
    '''
    print('Creating synthetic results for: ', repository)
    num_runs = 0
    while run_limit is None or num_runs >= run_limit:
        print('Creating TestRun #{}'.format(num_runs))
        create_synthetic_test_result(repository)
        num_runs += 1


def create_synthetic_test_result(repository: Repository) -> TestRun:
    '''
    Creates synthetic changes to a code base.  It first alters a repository's
    code base, creates a commit, and then runs the tests to see how the changes
    impacted the test results

    @param repository: the code base we are changing
    '''
    if not is_repo_clean(repository):
        msg = ('You attempted to work on an unclean repository.  Please run: \n'
               '"git checkout ." to clean the library')
        raise BugBuddyError(msg)
    edit_random_function(repository)
    commit = create_commit(repository)
    test_run = run_test(repository, commit)
    revert_commit(repository)


def edit_random_function(repository):
    '''
    Alters the repository in a very simplistic manner.  For right now, we are
    just going to take a method or function and add either an assert False or
    assert True to it

    @param repository: the code base we are changing
    '''
    # collect all the files

    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    # choose one file
    chosen_file = repo_files[random.randint(0, len(repo_files))]

    function_header_regex = 'def \w*\((\w|\s|,|=|\r|\*|\n|^\))*\):'
    chosen_file

