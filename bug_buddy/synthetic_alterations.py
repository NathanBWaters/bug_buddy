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
import inspect
import importlib
from importlib.machinery import SourceFileLoader
import os
import random
import re
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE
from bug_buddy.errors import BugBuddyError
from bug_buddy.execution import run_test
from bug_buddy.git_utils import (is_repo_clean,
                                 create_commit,
                                 revert_commit)
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, TestRun


ROUTINE_SIG = re.compile('def \w*\((\w|\s|,|=|\r|\*|\n|^\))*\):')



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
    # contains the methods/functions across the files
    routines = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        routines.extend(get_routines_from_file(repository, repo_file))

    seleted_routine = routines[random.randint(0, len(routines) - 1)]
    _add_assert_to_routine(seleted_routine)


def get_routines_from_file(repository, repo_file):
    '''
    Returns the methods and functions from the file
    '''
    # first load the entire project
    # root_module = os.path.join(repository.src_path, '__init__.py')
    # spec = importlib.util.spec_from_file_location(repository.name, root_module)
    # module = importlib.util.module_from_spec(spec)
    # sys.modules[spec.name] = module
    # spec.loader.exec_module(module)

    # now load the submodule
    submodule_name = repo_file.split('/')[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(
        '{}.{}'.format(repository.name, submodule_name),
        repo_file)
    repo_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repo_module)

    routines = []

    for member_name, member in inspect.getmembers(repo_module):
        try:
            if member_name == '__builtins__':
                continue

            # get functions
            if inspect.isfunction(member):
                print('adding one')
                routines.append(member)

            # TODO: this should be recursive because you can have nested classes
            #       even though that is rare
            if inspect.isclass(member):
                for class_member_name, class_member in inspect.getmembers(member):
                    # get methods from classes.

                    # TODO: I also need to add properties..
                    # isinstance(class_member, property)
                    # unfortunately it does not have a module
                    if (inspect.ismethod(class_member) or
                            inspect.isfunction(class_member)):
                        print('adding one')
                        routines.append(class_member)

        except Exception as e:
            logger.error('Hit error working on {} with: {}'
                         .format(member_name, e))

    file_routines = [routine for routine in routines if
                     inspect.getsourcefile(routine) == repo_file]

    return list(set(file_routines))


def _add_assert_to_routine(routine):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    routine_src_code, starting_line = inspect.getsourcelines(routine)
    repo_file = inspect.getsourcefile(routine)
    with open(repo_file, 'a') as file:
        file.seek(starting_line + len(routine_src_code))
        import pdb; pdb.set_trace()
        print('hmm')
