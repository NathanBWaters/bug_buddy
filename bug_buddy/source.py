'''
Code for interacting with the source code
'''
import ast
import os
import random
from typing import List
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE, DIFF_ADDITION
from bug_buddy.db import Session, get_all, create
from bug_buddy.errors import UserError, BugBuddyError
from bug_buddy.git_utils import get_diffs
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Function, FunctionHistory, Repository
from bug_buddy.schema.aliases import FunctionList, DiffList


def edit_functions(repository: Repository,
                   message=None,
                   get_message_func=None,
                   num_edits=None):
    '''
    Alters the repository in a very simplistic manner.  For right now, we are
    just going to take a method or function and add either an assert False or
    assert True to it

    @param repository: the code base we are changing
    @param commit: the currently empty commit we'll be adding changes to
    @param message: the string you want to add
    @param get_message_func: the function to call for getting the message
    @param num_edits: the number of edits you want to make.  Defaults to the
                      number of functions
    '''
    if not message and not get_message_func:
        raise UserError('You must either specify message or get_message_func '
                        'for synthetic_alterations.edit_functions')

    # contains the methods/functions across the files
    uneditted_functions = get_functions_from_repo(repository)

    # edit all functions with the message if not specified
    num_edits = num_edits or len(uneditted_functions)

    altered_functions = []

    for i in range(num_edits):
        function_index = random.randint(0, len(uneditted_functions) - 1)
        selected_function = uneditted_functions[function_index]

        # Debugging hackery
        # message = 'print("{} @ {} in {}")'.format(
        #     selected_function.node.name,
        #     selected_function.node.lineno,
        #     selected_function.file)
        if get_message_func:
            message = get_message_func(selected_function)

        selected_function.prepend_statement(message)

        altered_functions.append(selected_function)

        # the file has been editted.  This means we need to refresh the functions
        # with the correct line numbers.  However, we still don't want to edit
        # the function that we just previously altered.
        uneditted_functions = get_functions_from_repo(repository)

        for altered_function in altered_functions:
            matching_functions = [
                function for function in uneditted_functions
                if function.node.name == altered_function.node.name]

            closest_function = matching_functions[0]
            for matching_function in matching_functions:
                if (abs(matching_function.node.lineno - altered_function.node.lineno) <
                        abs(closest_function.node.lineno - altered_function.node.lineno)):
                    # we have found a function that is more likely to correspond
                    # with the original altered_function.
                    closest_function = matching_function

            # delete the already altered function from the list of available
            # functions
            uneditted_functions.remove(closest_function)


def get_functions_from_repo(repository: Repository):
    '''
    Returns the functions from the repository src files
    '''
    functions = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        functions.extend(get_functions_from_file(repository, repo_file))

    return functions


def get_functions_from_file(repository: Repository, repo_file: str):
    '''
    Returns the functions from the file.  They're created in the database if
    do not already exist
    '''
    functions = []

    with open(repo_file) as file:
        repo_file_content = file.read()
        repo_module = ast.parse(repo_file_content)
        for node in ast.walk(repo_module):
            if isinstance(node, ast.FunctionDef):
                relative_file_path = os.path.relpath(repo_file, repository.path)
                logger.info('Currently have node "{}" at path "{}@{}"'
                            .format(node.name, relative_file_path, node.lineno))
                function = get_function(
                    repository=repository,
                    node=node,
                    file_path=relative_file_path,
                )
                logger.info('Got function {}'.format(function))

                functions.append(function)

    return functions


def get_altered_functions(repository: Repository,
                          commit: Commit,
                          functions: FunctionList=[]) -> FunctionList:
    '''
    Given a repository and commit, return the functions that were altered.

    @repository: the repository that we are retrieving the functions from
    @commit: the commit that we are retrieving functions from
    @functions: if we already have the functions, we can pass them in so we
                do not have to retrieve the functions twice from source code
    '''
    if not functions:
        functions = get_functions_from_repo(repository)

    diffs = get_diffs(repository, commit)

    altered_functions = []
    for diff in diffs:
        # To find the corresponding function for the diff, we need to first
        # limit it to the same file and then see which functions encompass that
        # diff using line numbers.
        matching_functions = []
        for function in functions:
            if (function.file_path == diff.file_path and
                    function.first_line <= diff.first_line and
                    function.last_line >= diff.last_line):
                matching_functions.append(function)

        if not matching_functions:
            raise BugBuddyError('No matching function for diff: {}'.format(diff))

        # there's a change that multiple functions encompass the diff if there
        # are nested functions.  So we use the function that most tightly
        # encompasses the diff
        matching_function = None
        tighest_fit = sys.maxsize
        for function in matching_functions:
            if ((diff.line_number - function.first_line +
                 function.last_line - diff.line_number) < tighest_fit):
                matching_function = function

        altered_functions.append(matching_function)

    return altered_functions


def get_function(repository: Repository,
                 node,
                 file_path: str,
                 commit: Commit=None,
                 diffs: DiffList=[],
                 ):
    '''
    Given a function's ast node information, return a corresponding Function.
    If it already exists in the database, use that one.  Otherwise make a new
    instance but do not save it in the DB. We make sure the function in the
    database and the AST node are the same by tracking line numbers in the
    database and looking at the diff.

    For example:
        - If the file was not editted and the node's line number is the same
          as the Function's latest FunctionHistory line number, then we have
          found the corresponding Function instance for the node
        - If the file was editted, we need to see how many lines were added or
          subtracted above the Function's latest FunctionHistory line number.

        - Since we just retrieved the node from the source code, the node has
          each Function's current line number, which we will store as the
          FunctionHistory line_number

    @param repository: the Repository instance we are getting a function from
    @node: the corresponding AST Function node
    @file_path: the relative file path of the function
    @commit (optional): the commit this function is a part of.  Necessary for
                        if we need the diff
    '''
    session = Session.object_session(repository)

    potential_matching_functions = get_all(
        session,
        Function,
        repository=repository,
        name=node.name,
        file_path=file_path)

    # we have found a possible matching function.  We now need to make sure
    # that we have the correct matching function by comparing the line numbers
    if potential_matching_functions:
        if not diffs:
            diffs = get_diffs(repository, commit)

        # We have at least one matching function, now we make sure it's the
        # exact corresponding function but making sure the lines are the same.

        # First, we check to see if there are any diffs in the file that would
        # influence the location of this function. If there are not, then the
        # AST Function node should have the exact same line number as the
        # matching function's latest FunctionHistory.line_number.
        affecting_diffs = [
            diff for diff in diffs if file_path == diff.file_path and
            diff.first_line <= node.lineno
        ]

        diff_impact = 0
        if affecting_diffs:
            # The file was altered, so we now need to know how many lines were
            # added or subtracted above the function.
            diff_impact = sum([diff.size_difference for diff in affecting_diffs])
            logger.info('The following diffs are adding "{}" lines: {}'
                        .format(diff_impact, diffs))

        for function in potential_matching_functions:
            # if the function does not have node history, then that means it
            # is a new function that has not gone through the snapshot process
            if not function.function_history:
                continue

            if node.lineno == function.latest_history.first_line + diff_impact:
                function.node = node
                logger.info('We found the corresponding Function instance: {}'
                            .format(function))
                return function

        import pdb; pdb.set_trace()
        logger.info('The following potential_matching_functions did not match  '
                    'the node line number "{lineno}": {functions}'
                    .format(lineno=node.lineno,
                            functions=potential_matching_functions))

    # There are no perfectly matching fucntions so we therefore will return
    # a new Function instance. We do not store the function at this point
    new_function = create(
        session,
        Function,
        repository=repository,
        node=node,
        file_path=file_path)

    logger.info('There is a new function: {}'.format(new_function))
    return new_function
