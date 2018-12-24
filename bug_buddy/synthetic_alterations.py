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
from importlib.machinery import SourceFileLoader
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
    function_header_regex = re.compile('def \w*\((\w|\s|,|=|\r|\*|\n|^\))*\):')

    # contains the matches across the files
    matches_dict = {}

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        file_functions = get_functions_from_file(repo_file)
        with open(repo_file) as current_file:
            file_content = current_file.read()
            function_header_matches = re.finditer(
                function_header_regex,
                file_content)
            for match in function_header_matches:
                key = ('{file}|{starting}|{ending}'
                       .format(file=repo_file,
                               starting=match.start(),
                               ending=match.end()))
                matches_dict[key] = (match, repo_file)

    possible_matches = list(matches_dict.keys())
    import pdb; pdb.set_trace()
    num_methods = len(possible_matches)
    seleted_method = possible_matches[random.randint(0, num_methods)]
    match, file = matches_dict[seleted_method]
    is_innocuous_change = _add_assert_to_function(match, file)


def get_routines_from_file(repo_file):
    '''
    Returns the methods and functions from the file
    '''
    repo_module = SourceFileLoader(repo_file.split('/')[-1], repo_file).load_module()

    routines = []

    for member_name, member in inspect.getmembers(repo_module):
        if member_name == '__builtins__':
            continue

        # get functions
        if inspect.isfunction(member):
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
                    routines.append(class_member)

    routines = [routine for routine in routines if
                inspect.getmodule(routine) == repo_module]
    import pdb; pdb.set_trace()
    return list(set(routines))


def _add_assert_to_function(match, repo_file):
    '''
    Adds either a assert True or assert False right after the beginning to a
    method.  Returns whether the change was innocuous or not.
    '''
    with open(repo_file) as file:
        file_content = file.read()

        beginning_index = match.start - 1
        indent_count = 0
        # while file_content[beginning_index] != '\n':
