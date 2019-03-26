'''
Git utility methods
'''
import difflib
from git import Repo
from git.cmd import Git
import os
import subprocess
from typing import List

from bug_buddy.schema.aliases import DiffList
from bug_buddy.constants import (
    DEVELOPER_CHANGE,
    SYNTHETIC_RESET_CHANGE)
from bug_buddy.db import create, Session, get, get_or_create_diff
from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Diff, Repository, Function


def db_and_git_match(repository: Repository, command: str, log=True):
    '''
    Confirms if the database commit history and git commit history match.
    '''
    assert False, 'implement me!'


def run_cmd(repository: Repository, command: str, log=True):
    '''
    Runs a shell command
    '''
    if log:
        logger.info(command)

    process = subprocess.Popen(
        command,
        shell=True,
        cwd=repository.path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if log and stderr:
        if stdout:
            logger.info('stdout from command: "{}"\n{}'.format(command, stdout))
        if stderr:
            logger.error('stderr from command: "{}"\n{}'.format(command, stderr))

    return stdout.decode("utf-8").strip(), stderr.decode("utf-8").strip()


def is_repo_clean(repository: Repository, path=None):
    '''
    Asserts that the repository is clean, meaning there are no changes between
    the working tree and the HEAD

    https://stackoverflow.com/a/1587952
    '''
    path = path or repository.path
    cmd = ['git', 'diff', 'HEAD', '--exit-code']
    child = subprocess.Popen(cmd,
                             cwd=path,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    child.communicate()
    return_code = child.poll()
    return return_code == 0


def set_bug_buddy_branch(repository: Repository):
    '''
    All work should go to a defined git branch named "bug_buddy".  Idempotent.
    https://stackoverflow.com/a/35683029/4447761
    '''
    command = 'git checkout bug_buddy || git checkout -b bug_buddy'
    stdout, stderr = run_cmd(repository, command, log=False)

    # make sure the branch is pushed remotely
    all_branches = Git(repository.path).branch('-a')

    if 'remotes/origin/bug_buddy' not in all_branches:
        command = 'git push --set-upstream origin bug_buddy'
        stdout, stderr = run_cmd(repository, command, log=False)


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
                  commit_type: str=DEVELOPER_CHANGE,
                  allow_empty=False) -> Commit:
    '''
    Given a repository, create a commit

    @param repository: the repository to be analyzed
    @param name: the commit name/message
    @param commit_type: metadata about the commit
    @param empty: whether or not the commit is empty
    '''
    commit_name = name or 'bug_buddy_commit'
    Git(repository.path).add('-A')

    # You should only create commits on the "bug_buddy" branch

    Git(repository.path).commit(
        '-m "{commit_name}"'.format(commit_name=commit_name),
        '--allow-empty' if allow_empty else None)

    set_bug_buddy_branch(repository)

    commit_id = get_commit_id(repository)
    return _store_commit(repository, commit_id, commit_type)


def _store_commit(repository: Repository,
                  commit_id: str,
                  commit_type: str=DEVELOPER_CHANGE) -> Commit:
    '''
    Stores the commit in the database
    '''
    session = Session.object_session(repository)

    branch = get_branch_name(repository)

    commit = create(session,
                    Commit,
                    repository=repository,
                    commit_id=commit_id,
                    branch=branch,
                    commit_type=commit_type)
    logger.info('Created commit: {}'.format(commit))

    return commit


def update_commit(repository: Repository):
    '''
    Updates the commit with the altered local space

    @param repository: the repository with the commit to be updated
    '''
    Git(repository.path).add('-A')

    # git commit --amend
    Git(repository.path).commit('--amend')


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


def get_most_recent_commit(repository: Repository,
                           branch='bug_buddy',
                           create=False) -> Commit:
    '''
    Returns the latest commit from the specified branch.  Defaults to the latest
    in master
    '''
    set_bug_buddy_branch(repository)

    session = Session.object_session(repository)
    command = 'git log {branch} --format="%H" | head -1'.format(branch=branch)

    commit_id, stderr = run_cmd(repository, command)

    commit = get(session, Commit, commit_id=commit_id)

    # if not commit and not create:
    #     msg = ('Requested the most recent commit, but commit {} is not in the '
    #            'database and create is set to False'.format(commit_id))
    #     raise BugBuddyError(msg)

    # if the most recent commit does not already exist in the database and
    # create is set to True, store the commit
    if not commit and create:
        commit = _store_commit(repository, commit_id)

    return commit


def get_commits_only_in_branch(repository, branch='origin/bug_buddy') -> List[str]:
    '''
    Returns the commits that are in one branch such as the bug_buddy branch

        $ git log origin/master..origin/bug_buddy --format="%H"
        > 8ce7ced9a5fe1f0b7e2cf3df1ce33b3eadc62d95
        > 48a19ebc48a5c86f2aa73ffea4fe60141bf73520
        > 80c9bbe863a077e69a37d0ef560f2553ae622ee3
        > c3b36165440c53ab5e73de88819beb3ca96b3134

    @returns: list of commit strings. For example:
        [
            '8ce7ced9a5fe1f0b7e2cf3df1ce33b3eadc62d95',
            '48a19ebc48a5c86f2aa73ffea4fe60141bf73520',
            '80c9bbe863a077e69a37d0ef560f2553ae622ee3',
            'c3b36165440c53ab5e73de88819beb3ca96b3134'
        ]
    '''
    command = 'git log origin/master..{} --format="%H"'.format(branch)
    stdout, stderr = run_cmd(repository, command)
    return stdout.split('\n')


def get_previous_commit(commit: Commit):
    '''
    Returns the commit id created right before the provided commit
    '''
    previous_commit_id = get_previous_commit_id(commit)
    return get(Session.object_session(commit), Commit, commit_id=previous_commit_id)


def get_previous_commit_id(commit: Commit):
    '''
    Returns the commit id created right before the provided commit
    '''
    command = 'git show --q --format="%H" {}^1'.format(commit.commit_id)
    stdout, stderr = run_cmd(commit.repository, command)
    return stdout.split('\n')[0]


def revert_to_master(repository: Repository):
    '''
    Creates a new commit which contains the same content as a fresh branch
    without any of it's previous edits.

    The easiest way to do this is by simply resetting all commits in the
    bug_buddy branch and create a new commit out of that
    '''
    logger.info('Starting to create reset commit')
    set_bug_buddy_branch(repository)

    # create the changes that effectively reverts all work we have done on this
    # branch
    run_cmd(repository, 'git checkout master {}'.format(repository.path))
    run_cmd(repository, 'git reset {}'.format(repository.path))

    return


def create_reset_commit(repository: Repository):
    '''
    Resets the branch, and then creates a commit out of that change
    '''
    revert_to_master(repository)

    # it's possible that nothing has been changed after the git revert.
    # For example, if the initial synthetic commit only added 'assert False'
    # and then they were all undone. If so, then trying to create a commit
    # when nothing is altered/staged would error.  Therefore, we have to make
    # sure the repo is dirty before we make a commit.
    if not is_repo_clean(repository):
        create_commit(repository,
                      'reset_commit',
                      commit_type=SYNTHETIC_RESET_CHANGE)
    else:
        logger.info('Did not need to create reset commit since the repo is '
                    'clean.')


def delete_bug_buddy_branch(repository):
    '''
    Deletes the bug buddy branch
    '''
    # switch to master so we can delete blame buddy branch
    Git(repository.path).checkout('master')

    # this deletes the branch in the remote server
    command = 'git push origin --delete bug_buddy'
    stdout, stderr = run_cmd(repository, command)

    # delete the branch locally
    command = 'git branch -D bug_buddy'
    stdout, stderr = run_cmd(repository, command)


def clone_repository(repository, path):
    '''
    Clones a repository
    '''
    if not os.path.exists(path):
        os.makedirs(path)

    command = 'git clone {url} {path}'.format(url=repository.url, path=path)
    run_cmd(repository, command)
