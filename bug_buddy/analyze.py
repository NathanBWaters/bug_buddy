'''
API for anlyzing a code repository at a particular commit
'''
from bug_buddy.schema import Repository, Commit


def analyze_repository(repository: Repository, commit: Commit) -> dict:
    '''
    Analyze a repository for a given commit.  The results are saved to the
    database.

    @param repository: the repository to be analyzed
    @param commit: the commit to be analyzed
    '''
    