'''
Git utility methods
'''
from git import Repo
from git.cmd import Git
import subprocess

from bug_buddy.db import create, Session
from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
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
    command = 'git checkout bug_buddy || git checkout -b bug_buddy'
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=repository.path)
    # stdout=subprocess.PIPE,
    # stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # make sure the branch is pushed remotely
    all_branches = Git(repository.path).branch('-a')

    if 'remotes/origin/bug_buddy' not in all_branches:
        command = 'git push --set-upstream origin bug_buddy'
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=repository.path)
        stdout, stderr = process.communicate()


def get_commit_id(repository: Repository) -> str:
    '''
    Given a repository, return the branch name
    '''
    commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                        cwd=repository.path)
    # convert bytes to string and take away newline
    return commit_id.decode("utf-8").strip()


def get_branch_name(repository: Repository) -> str:
    '''
    Given a repository, return the branch name
    '''
    branch_name = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        cwd=repository.path)
    # convert bytes to string and take away newline
    return branch_name.decode("utf-8").strip()


def create_commit(repository: Repository,
                  name: str=None,
                  is_synthetic: bool=False) -> Commit:
    '''
    Given a repository, create a commit

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    commit_name = name or 'bug_buddy_synthetic_commit'
    # You should only create commits on the "bug_buddy" branch
    set_bug_buddy_branch(repository)

    Git(repository.path).add('-A')
    Git(repository.path).commit('-m', commit_name)

    commit_id = get_commit_id(repository)
    branch = get_branch_name(repository)

    session = Session.object_session(repository)
    commit = create(session,
                    Commit,
                    repository=repository,
                    commit_id=commit_id,
                    branch=branch,
                    is_synthetic=is_synthetic)
    logger.info('Created commit: {}'.format(commit))
    return commit


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


def delete_bug_buddy_branch(repository):
    '''
    Deletes the bug buddy branch
    '''
    # switch to master so we can delete blame buddy branch
    Git(repository.path).checkout('master')

    # this deletes the branch in the remote server
    command = 'git push origin --delete bug_buddy'
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=repository.path)
    stdout, stderr = process.communicate()

    # delete the branch locally
    command = 'git branch -D bug_buddy'
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=repository.path)
    stdout, stderr = process.communicate()
