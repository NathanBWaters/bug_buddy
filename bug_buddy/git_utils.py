'''
Git utility methods
'''
import difflib
from git import Repo
from git.cmd import Git
import os
import re
import subprocess
import tempfile
from typing import List
import whatthepatch

from bug_buddy.schema.aliases import DiffList
from bug_buddy.constants import (
    DEVELOPER_CHANGE,
    SYNTHETIC_RESET_CHANGE)
from bug_buddy.db import create, Session, get, get_or_create_diff
from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema import Commit, Diff, Repository

RANGE_INFO_REGEX = '@@ -?\d+,\d+ \+\d+,\d+ @@'


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

    # You should only create commits on the "bug_buddy" branch
    set_bug_buddy_branch(repository)

    Git(repository.path).add('-A')

    Git(repository.path).commit(
        '-m "{commit_name}" {allow_empty}'
        .format(commit_name=commit_name,
                allow_empty='--allow-empty' if allow_empty else ''))

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


def add_function_histories(repository: Repository, commit: Commit):
    '''
    Adds the FunctionHistory for each function in the commit.  It adds
    information such as whether or not it was changed.
    '''
    assert False, 'implement add_function_histories'


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


def revert_diff(diff: Diff):
    '''
    Reverts the diff from the source
    '''
    try:
        file_name = 'bugbuddy_diff_id_{}'.format(diff.id)
        suffix = '.patch'
        temp_output = tempfile.NamedTemporaryFile(
            prefix=file_name,
            suffix=suffix,
            dir=diff.repository.path)
        temp_output.write(str.encode(diff.patch))
        temp_output.seek(0)
        temp_output.read()

        command = 'git apply -R {file_path}'.format(file_path=temp_output.name)

        stdout, stderr = run_cmd(diff.repository, command)
        if stderr:
            msg = ('Error trying to revert diff {diff} with patch:\n{patch}\n\n'
                   'stderr: {stderr}'
                   .format(diff=diff, patch=diff.patch, stderr=stderr))
            raise BugBuddyError(msg)
    finally:
        temp_output.close()


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

    if not commit and not create:
        msg = ('Requested the most recent commit, but commit {} is not in the '
               'database and create is set to False'.format(commit_id))
        raise BugBuddyError(msg)

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


def get_patches_from_diffs(repository: Repository,
                           commit: Commit=None,
                           split_per_method=True) -> List[str]:
    '''
    Creates a patch file containing all the diffs in the repository and then
    returns all those patches as a list of patches
    '''
    # this command will output the diff information into stdout
    command = 'git --no-pager diff'
    diff_data, _ = run_cmd(repository, command)

    patches = diff_data.split('diff --git ')[1:]
    if split_per_method:
        method_granular_patches = []
        for patch in patches:
            if len(re.findall(RANGE_INFO_REGEX, patch)) > 1:
                # it looks like there were multiple edits that made it into
                # the same hunk.  We need to split each part of the patch hunk
                # into it's own chunk.  First step is to keep the first four
                # lines, the header, which will become the first four lines
                # of each sub-chunk.  For example:
                patch_lines = patch.split('\n')
                header = patch_lines[0:4]

                starting_line = 4
                sub_patch_lines = []
                for i in range(4, len(patch_lines)):
                    if (re.findall(RANGE_INFO_REGEX, patch_lines[i]) or
                            i == len(patch_lines) - 1):
                        if sub_patch_lines:
                            sub_patch = '\n'.join(
                                header + patch_lines[starting_line: i])
                            method_granular_patches.append(sub_patch)

                            # start over for the next subpatch
                            sub_patch_lines = []
                            starting_line = i

                    sub_patch_lines.append(patch_lines[i])

            else:
                method_granular_patches.append(patch)
        patches = method_granular_patches
    return patches


def get_diffs(repository: Repository, commit: Commit=None) -> DiffList:
    '''
    Returns a list of diffs from a repository
    '''
    session = Session.object_session(repository)
    if not commit:
        commit = get_most_recent_commit(repository)

    diffs = []

    patches = get_patches_from_diffs(repository, commit)
    for patch in patches:
        patch = list(whatthepatch.parse_patch(patch))[0]

        # this only works for addition diffs
        first_line = 0
        for old_line_number, new_line_number, line in patch.changes:
            if not old_line_number and new_line_number:
                first_line = new_line_number
                break

        diff = get_or_create_diff(
            session=session,
            commit=commit,
            first_line=first_line,
            last_line=first_line + 1,
            file_path=patch.header.new_path,
            patch=patch.text)

        diffs.append(diff)

    return diffs
