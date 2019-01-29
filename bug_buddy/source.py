'''
Code for interacting with the source code
'''
import ast
import random
from typing import List

from bug_buddy.constants import PYTHON_FILE_TYPE
from bug_buddy.errors import UserError
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Routine


def edit_routines(repository: Repository,
                  message=None,
                  get_message_func=None,
                  num_edits=None):
    '''
    Alters the repository in a very simplistic manner.  For right now, we are
    just going to take a method or function and add either an assert False or
    assert True to it

    @param repository: the code base we are changing
    @param message: the string you want to add
    @param get_message_func: the function to call for getting the message
    @param num_edits: the number of edits you want to make.  Defaults to the
                      number of routines
    '''
    if not message and not get_message_func:
        raise UserError('You must either specify message or get_message_func '
                        'for synthetic_alterations.edit_routines')

    # contains the methods/functions across the files
    uneditted_routines = get_routines_from_repo(repository)

    # edit all routines with the message if not specified
    num_edits = num_edits or len(uneditted_routines)

    altered_routines = []

    for i in range(num_edits):
        routine_index = random.randint(0, len(uneditted_routines) - 1)
        selected_routine = uneditted_routines[routine_index]

        # Debugging hackery
        # message = 'print("{} @ {} in {}")'.format(
        #     selected_routine.node.name,
        #     selected_routine.node.lineno,
        #     selected_routine.file)
        if get_message_func:
            message = get_message_func(selected_routine)

        selected_routine.prepend_statement(message)

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


def get_routines_from_repo(repository: Repository):
    '''
    Returns the routines from the repository src files
    '''
    routines = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        routines.extend(get_routines_from_file(repository, repo_file))

    return routines


def get_routines_from_file(repository: Repository, repo_file: str):
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


def add_lines(repository: Repository):
    '''
    Adds all the lines of a repository to the database

    @param repository: the repository
    '''
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)
    for repo_file in repo_files:
        logger.info('Importing lines from: "{}"'.format(repo_file))
        with open(repo_file) as file:
            repo_file_content = file.read()
            ast_representation = ast.parse(repo_file_content)
            repo_file_content = repo_file_content.split('\n')
            # import pdb; pdb.set_trace()
            for node in ast.walk(ast_representation):
                if hasattr(node, 'lineno'):
                    line_content = repo_file_content[node.lineno - 1]
                    print('content: "{}"'.format(line_content))
                    print('node: ', node)
                    print('column_offset: ', node.col_offset)
                    try:
                        line_ast = ast.parse(line_content)
                        print('line_ast: ', line_ast)
                    except Exception as e:
                        print('Failed to parse line with error {}'.format(e))

        break

