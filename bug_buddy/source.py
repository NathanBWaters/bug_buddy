'''
Code for interacting with the source code
'''
import ast
import os
import random
from typing import List

from bug_buddy.constants import PYTHON_FILE_TYPE, DIFF_ADDITION
from bug_buddy.db import get_or_create_function, create, Session
from bug_buddy.errors import UserError
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Function, Repository


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

    session = Session.object_session(repository)
    with open(repo_file) as file:
        repo_file_content = file.read()
        repo_module = ast.parse(repo_file_content)
        for node in ast.walk(repo_module):
            if isinstance(node, ast.FunctionDef):
                relative_file_path = os.path.relpath(repo_file, repository.path)

                function = get_or_create_function(
                    session,
                    repository=repository,
                    node=node,
                    file_path=relative_file_path,
                )
                functions.append(function)

    return functions
