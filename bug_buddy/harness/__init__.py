'''
Commands for running tests and recording the output
'''
from bug_buddy.schema import Repository, TestRun, Commit


def run_test(repository: Repository, commit: Commit):
    '''
    Runs a repository's tests and records the results
    '''
    print('Implement run_test')
    assert False


def record_test_results(repository: Repository, test_run: TestRun) -> dict:
    '''
    Analyze a repository for a given commit.  The results are saved to the
    database.

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    print('Implement run_test')
    assert False
