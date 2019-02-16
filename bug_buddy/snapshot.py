'''
Given a repository and commit, it will store the change data into the database.
'''
import os
import random
from typing import List
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE
from bug_buddy.db import create, Session, session_manager
from bug_buddy.errors import UserError, BugBuddyError
from bug_buddy.git_utils import create_diffs
from bug_buddy.logger import logger
from bug_buddy.source import (get_functions_from_repo,
                              create_synthetic_alterations)
from bug_buddy.schema import (Commit,
                              DiffCommitLink,
                              Function,
                              FunctionHistory,
                              FunctionToTestLink,
                              Repository,
                              TestRun)
from bug_buddy.schema.aliases import FunctionList, DiffList


def snapshot_commit(repository: Repository, commit: Commit):
    '''
    Given a repository and commit, store the necessary data such as the
    Functions, FunctionHistory, and Diff instances.

    It will do this in the following order:
        - Get the diffs
        - Get the functions
        - Store new functions
        - Create new FunctionHistory for all functions that exist
        - Store the diffs with their corresponding FunctionHistory
    '''
    logger.info('Snapshotting commit {}'.format(commit))
    session = Session.object_session(repository)

    # retrieves the functions
    functions = get_functions_from_repo(repository, commit)
    session.add_all(functions)

    # create FunctionHistory instances for each Function
    diffs = create_diffs(repository, commit)
    save_function_histories(repository, commit, functions, diffs)

    # save the diffs
    save_diffs(repository, commit, diffs)


def snapshot_initialization(repository: Repository):
    '''
    Initializes the repository
    '''
    # Adds a random number of edits to the repository.
    if not repository.diffs:
        # adds 'assert False' to each function
        logger.info('Creating synthetic alterations')
        create_synthetic_alterations(repository)
    else:
        logger.info('Already initialized')


def snapshot_diff_commit_link(commit: Commit, diff_set: DiffList):
    '''
    Creates the mapping between the commits and the diff_set
    '''
    session = Session.object_session(commit)
    for diff in diff_set:
        create(session,
               DiffCommitLink,
               commit=commit,
               diff=diff)


def save_function_histories(repository: Repository,
                            commit: Commit,
                            functions: FunctionList,
                            diffs: DiffList):
    '''
    Creates a FunctionHistory instance for each function
    '''
    session = Session.object_session(repository)
    for function in functions:

        was_altered = any(
            [diff.first_line >= function.first_line and
             diff.last_line <= function.last_line and
             diff.file_path == function.file_path
             for diff in diffs])

        create(
            session,
            FunctionHistory,
            function=function,
            commit=commit,
            first_line=function.first_line,
            last_line=function.last_line,
            altered=was_altered)


def save_diffs(repository: Repository,
               commit: Commit,
               diffs: DiffList):
    '''
    Saves the diffs
    '''
    session = Session.object_session(repository)
    for diff in diffs:
        diff.commit = commit

    session.add_all(diffs)
