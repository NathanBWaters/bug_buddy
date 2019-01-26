'''
Git utility methods
'''
import difflib
from git import Repo
from git.cmd import Git
import re
import subprocess
from typing import List

from bug_buddy.constants import DEVELOPER_CHANGE, SYNTHETIC_RESET_CHANGE
from bug_buddy.db import create, Session
from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, Commit, Diff


def run_cmd(repository: Repository, command: str, log=False):
    '''
    Runs a shell command
    '''
    logger.info('Running shell command: "{}"'.format(command))
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=repository.path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if log:
        if stdout:
            logger.info(stdout)
        if stderr:
            logger.error(stderr)

    return stdout.decode("utf-8").strip(), stderr.decode("utf-8").strip()


def is_repo_clean(repository: Repository):
    '''
    Asserts that the repository is clean
    https://stackoverflow.com/a/45989092/4447761
    '''
    cmd = ['git', 'diff', '--exit-code']
    child = subprocess.Popen(cmd,
                             cwd=repository.path,
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
    stdout, stderr = run_cmd(repository, command)

    # make sure the branch is pushed remotely
    all_branches = Git(repository.path).branch('-a')

    if 'remotes/origin/bug_buddy' not in all_branches:
        command = 'git push --set-upstream origin bug_buddy'
        stdout, stderr = run_cmd(repository, command)


def get_diffs(repository: Repository,
              starting_commit_id: str,
              ending_commit_id: str):
    '''
    Returns a list of Diff instances.  This is still under the assumption that
    we have simply added one line statements.
    '''
    repo = Repo(repository.path)

    starting_commit = repo.commit(starting_commit_id)
    ending_commit = repo.commit(ending_commit_id)

    diff_index = starting_commit.diff(ending_commit)

    diffs = []

    for diff_item in diff_index.iter_change_type('M'):
        file_at_start = diff_item.a_blob.data_stream.read().decode('utf-8').split('\n')
        file_at_end = diff_item.b_blob.data_stream.read().decode('utf-8').split('\n')

        diff = difflib.unified_diff(file_at_start,
                                    file_at_end,
                                    fromfile='file1',
                                    tofile='file2',
                                    lineterm='',
                                    n=0)
        lines = list(diff)
        for i in range(len(lines)):
            if lines[i].startswith('@@'):
                # the diff follows the pattern.
                #   '@@ previous_lineno, duration of addition, updated lineno'
                #   '@@ -1381,0 +1382 @@'
                #   '@@ -151,0 +152 @@'
                # For more information:
                #   https://www.wikiwand.com/en/Diff#/Unified_format
                range_information = lines[i]
                line_number = re.search('\+\d+', range_information).group()
                line_number = int(line_number[1:]) - 1

                added_content = lines[i + 1]

                diff = Diff(added_content, line_number, diff_item.a_path)

                diffs.append(diff)

    return diffs


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
                  commit_type: str=DEVELOPER_CHANGE) -> Commit:
    '''
    Given a repository, create a commit

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    commit_name = name or 'bug_buddy_synthetic_commit'
    # You should only create commits on the "bug_buddy" branch
    set_bug_buddy_branch(repository)

    Git(repository.path).add('-A')

    Git(repository.path).commit('-m "{}"'.format(commit_name))

    commit_id = get_commit_id(repository)
    branch = get_branch_name(repository)

    session = Session.object_session(repository)
    commit = create(session,
                    Commit,
                    repository=repository,
                    commit_id=commit_id,
                    branch=branch,
                    commit_type=commit_type)
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


def get_most_recent_commit(repository, branch='origin/master'):
    '''
    Returns the latest commit from the specified branch.  Defaults to the latest
    in master
    '''
    command = 'git log {branch} --format="%H" | head -1'.format(branch=branch)

    commit_id, stderr = run_cmd(repository, command)

    return commit_id


def get_commits_only_in_branch(repository, branch='origin/bug_buddy') -> List[str]:
    '''
    Returns the commits that are in one branch such as the bug_buddy branch

        $ git log origin/master..origin/bug_buddy --format="%H"
        > 8ce7ced9a5fe1f0b7e2cf3df1ce33b3eadc62d95
        > 48a19ebc48a5c86f2aa73ffea4fe60141bf73520
        > 80c9bbe863a077e69a37d0ef560f2553ae622ee3
        > c3b36165440c53ab5e73de88819beb3ca96b3134
    '''
    command = 'git log origin/master..{} --format="%H"'.format(branch)
    stdout, stderr = run_cmd(repository, command)
    return stdout.split('\n')


def create_reset_commit(repository: Repository):
    '''
    Creates a new commit which contains the same content as a fresh branch
    without any of it's previous edits.

    The easiest way to do this is by simply resetting all commits in the
    bug_buddy branch and create a new commit out of that
    '''
    logger.info('Starting to create reset commit')
    set_bug_buddy_branch(repository)

    bug_buddy_commits = get_commits_only_in_branch(repository,
                                                   branch='bug_buddy')
    bug_buddy_commits = ' '.join(bug_buddy_commits)

    # create the changes that effectively reverts all work we have done on this
    # branch
    command = ('git revert --no-commit {bug_buddy_commits}'
               .format(bug_buddy_commits=bug_buddy_commits))
    run_cmd(repository, command)

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
