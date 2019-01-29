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
import os
import random
import re
import sys
from typing import List

from bug_buddy.blaming import blame
from bug_buddy.constants import (BENIGN_STATEMENT,
                                 ERROR_STATEMENT,
                                 PYTHON_FILE_TYPE,
                                 SYNTHETIC_CHANGE,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import session_manager, Session
from bug_buddy.errors import UserError
from bug_buddy.git_utils import (get_most_recent_commit,
                                 is_repo_clean,
                                 create_commit,
                                 git_push,
                                 create_reset_commit,
                                 revert_unstaged_changes,
                                 run_cmd,
                                 set_bug_buddy_branch)
from bug_buddy.runner import run_test
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Routine, TestRun, Commit, Diff
from bug_buddy.source import edit_routines

# aliases for typing
DiffList = List[Diff]


def generate_synthetic_test_results(repository: Repository, run_limit: int):
    '''
    Creates multiple synthetic changes and test results
    '''
    num_runs = 0
    while run_limit is None or num_runs <= run_limit:
        with session_manager() as session:
            session.add(repository)
            logger.info('Creating TestRun #{}'.format(num_runs))

            # create an initial change, which asserts a random number of edits to
            # the repository
            commit = create_synthetic_alterations(repository)

            # run all tests against the synthetic change
            test_run = run_test(repository, commit)

            # determine which lines caused which test failures
            blame(repository, test_run)

            # revert to beginning of the branch, and then push that commit as a
            # new commit so we non-destructively can repeat this process with a
            # 'fresh' branch.
            create_reset_commit(repository)

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


def library_is_testable(repository):
    '''
    Returns whether or not the library is testable.  It does this by running
    pytest --collect-only.  If there's anything in the stderr than we are
    assuming we have altered a method that is called during import of the
    library.  This is a huge limitation of bug_buddy.
    '''
    command = 'pytest --collect-only'
    stdout, stderr = run_cmd(repository, command)
    if stderr:
        return False

    return True


def create_synthetic_alterations(repository: Repository):
    '''
    Creates synthetic changes to a code base, creates a commit, and then runs
    the tests to see how the changes impacted the test results.  These changes
    are either 'assert False' or 'assert True'.

    @param repository: the code base we are changing
    '''
    if not is_repo_clean(repository):
        msg = ('You attempted to work on an unclean repository.  Please run: \n'
               '"git checkout ." to clean the library')
        raise UserError(msg)

    # make sure we are on the bug buddy branch
    set_bug_buddy_branch(repository)

    # make synthetic alterations to the project
    num_edits = random.randint(1, int(len(repository.get_src_files()) / 4))
    edit_routines(repository,
                  get_message_func=_get_assert_statement,
                  num_edits=num_edits)

    # TODO: we want to make sure we can still import the library after we have
    # done the edits.  We definitely need a better way to do this.  This is
    # caused when simply importing the library hits a malign edit
    while(not library_is_testable(repository)):
        logger.info('Unable to test against the library with current edits. '
                    'Trying again.')
        revert_unstaged_changes(repository)
        edit_routines(repository,
                      get_message_func=_get_assert_statement,
                      num_edits=num_edits)

    session = Session.object_session(repository)

    commit = create_commit(repository,
                           name='synthetic_alteration_change',
                           commit_type=SYNTHETIC_CHANGE)

    # add the commit and test run to our database
    session.add(commit)
    return commit


def _get_assert_statement(routine):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    is_error_statement = random.randint(0, 1)
    return ERROR_STATEMENT if is_error_statement else BENIGN_STATEMENT
