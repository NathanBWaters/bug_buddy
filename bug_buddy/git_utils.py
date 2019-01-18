'''
Git utility methods
'''
from git import Repo
from git.cmd import Git
import subprocess

from bug_buddy.errors import BugBuddyError
from bug_buddy.schema import Repository, Commit


def is_repo_clean(repository: Repository):
    '''
    Asserts that the repository is clean
    https://stackoverflow.com/a/45989092/4447761
    '''
    cmd = ['git', 'diff', '--exit-code']
    child = subprocess.Popen(cmd, cwd=repository.path, stdout=subprocess.PIPE)
    child.communicate()
    return_code = child.poll()
    return return_code == 0


def set_bug_buddy_branch(repository: Repository):
    '''
    All work should go to a defined git branch named "bug_buddy".  Idempotent.
    https://stackoverflow.com/a/35683029/4447761
    '''
    subprocess.Popen(['git', 'checkout', 'bug_buddy', '||',
                      'git', 'checkout', '-b', 'bug_buddy'],
                     cwd=repository.path)


def create_commit(repository: Repository) -> Commit:
    '''
    Given a repository, create a commit

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    # You should only create commits on the "bug_buddy" branch
    set_bug_buddy_branch(repository)

    Git(repository.path).add('-A')
    Git(repository.path).commit('-m', 'bug_buddy_synthetic_commit')

    logger.info('Created commit: {}'.format(commit))


def revert_commit(repository: Repository, commit: Commit):
    '''
    Resets the repository, meaning it reverts back any edits that have been
    done on the bug_buddy branch to the original commit.  It then creates a new
    commit to push the reverted changes.  We don't save the revert commit.
    '''
    # You should only revert commits on the "bug_buddy" branch
    set_bug_buddy_branch(repository)

    Git(repository.path).revert(commit.commit_id)


def revert_unstaged_changes(repository: Repository):
    '''
    Reverts the un-staged changes in a repository
    '''
    Git(repository.path).checkout('.')


def git_push(repository: Repository):
    '''
    Pushes the commit to the "bug_buddy" branch
    '''
    Git(repository.path).push('--set-upstream', 'origin', 'bug_buddy')


def get_repository_name_from_url(url: str):
    '''
    Gets repository name given a url
    '''
    return url.split('/')[-1]


def get_repository_url_from_path(path: str):
    '''
    Gets the url of the repository given the path
    '''
    return Repo(path).remotes.origin.url

