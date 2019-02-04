'''
Given a repository and commit, it will store the change data into the database.
'''
import os
import random
from typing import List
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE, DIFF_ADDITION
from bug_buddy.db import create, Session, session_manager
from bug_buddy.errors import UserError, BugBuddyError
from bug_buddy.git_utils import get_diffs
from bug_buddy.logger import logger
from bug_buddy.source import get_functions_from_repo
from bug_buddy.schema import Commit, Function, FunctionHistory, Repository
from bug_buddy.schema.aliases import FunctionList, DiffList


def snapshot(repository: Repository, commit: Commit):
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
    session = Session.object_session(repository)

    # retrieves the functions
    functions = get_functions_from_repo(repository)
    session.add_all(functions)

    # create FunctionHistory instances for each Function
    diffs = get_diffs(repository, commit)
    save_function_histories(repository, commit, functions, diffs)

    save_diffs(repository, commit, diffs)


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
