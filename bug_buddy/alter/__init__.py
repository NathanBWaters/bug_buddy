'''
API for editing a repository and generating synthetic data
'''
import random

from bug_buddy.schema import Repository, TestRun
from bug_buddy.harness import run_test


def generate_synthetic_test_results(repository: Repository, run_limit: int=None):
    '''
    Creates multiple synthetic changes and test results
    '''
    num_runs = 0
    while run_limit is None or num_runs >= run_limit:
        print('Creating TestRun #{}'.format(num_runs))
        create_synthetic_test_result(repository)
        num_runs += 1
    print('repository: ', repository)



def create_synthetic_test_result(repository: Repository) -> TestRun:
    '''
    Creates synthetic changes to a code base.  It first alters a repository's
    code base, creates a commit, and then runs the tests to see how the changes
    impacted the test results

    @param repository: the code base we are changing

    '''
    repository.reset_repo()
    simple_edit(repository)
    commit = create_commit(repository)
    return run_test(repository, commit)


def simple_edit(repository):
    '''
    Alters the repository in a simple way.

    @param repository: the code base we are changing
    '''
    files = repository.get_files()
    num_edits = random.randint(0, 10)

    # for i in range(num_edits):
    print('Implement simple_edit')
    assert False


def create_commit(repository: Repository) -> dict:
    '''
    Given a repository that has edits, create a commit

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    pass
