'''
Code for interacting with the source code
'''
import ast
import os
import random
import re
import tempfile
import whatthepatch

from typing import List
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE, MIRROR_ROOT, SYNTHETIC_CHANGE
from bug_buddy.db import Session, get_all, create, delete, get
from bug_buddy.errors import UserError, BugBuddyError
from bug_buddy.git_utils import (
    clone_repository,
    create_commit,
    get_most_recent_commit,
    run_cmd)
from bug_buddy.logger import logger
from bug_buddy.runner import library_is_testable
from bug_buddy.schema import Commit, Diff, Function, FunctionHistory, Repository
from bug_buddy.schema.aliases import FunctionList, DiffList

RANGE_INFO_REGEX = '@@ -?\d+,\d+ \+\d+,\d+ @@'


class RewriteFunctions(ast.NodeTransformer):
    '''
    Used for transforming a repository
    '''
    def __init__(self,
                 repository: Repository,
                 commit: Commit,
                 file_path: str,
                 module,
                 prepend_assert_false=True):
        '''
        Creates a RewriteFunctions Ast Transformer
        '''
        self.repository = repository
        self.commit = commit
        self.prepend_assert_false = prepend_assert_false
        self.file_path = file_path
        self.module = module
        self.num_edits = 0

    def visit_FunctionDef(self, node):
        '''
        Is called when the transformer hits a function node
        '''
        if self.prepend_assert_false:
            self._prepend_assert_false(node)

    def _prepend_assert_false(self, node):
        '''
        Adds an 'assert False' to the node
        '''
        session = Session.object_session(self.repository)
        # There are no perfectly matching fucntions so we therefore will return
        # a new Function instance. We do not store the function at this point
        function = create(
            session,
            Function,
            repository=self.repository,
            node=node,
            file_path=self.file_path)

        logger.info('There is a new function: {}'.format(function))

        added_line = function.prepend_statement('assert False',
                                                offset=self.num_edits)

        if library_is_testable(self.repository):
            # create a new diff from this one change
            diffs = create_diffs(
                self.repository,
                commit=self.commit,
                function=function,
                is_synthetic=True)
            assert len(diffs) == 1
            diff = diffs[0]
            logger.info('Created diff: {}'.format(diff))

            # this is the function's synthetic diff
            function.synthetic_diff = diff

            # go back to a clean repository
            revert_diff(diff)

        else:
            # remove the addition from the source code
            function.remove_line(added_line)
        return node


def create_synthetic_alterations(repository: Repository):
    '''
    Creates synthetic changes to a code base, creates a commit, and then runs
    the tests to see how the changes impacted the test results.  These changes
    are either 'assert False' or 'assert True'.

    @param repository: the code base we are changing
    @param commit: the empty commit we're adding changes to
    '''
    # create an empty commit that the diffs will be added to
    commit = create_commit(
        repository,
        name='synthetic_alterations',
        commit_type=SYNTHETIC_CHANGE,
        allow_empty=True)
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for file_path in repo_files:
        file_module = get_module_from_file(file_path)
        transformer = RewriteFunctions(repository=repository,
                                       commit=commit,
                                       file_path=file_path,
                                       prepend_assert_false=True,
                                       module=file_module)
        transformer.visit(file_module)


def get_functions_from_repo(repository: Repository, commit: Commit=None):
    '''
    Returns the functions from the repository src files
    '''
    functions = []

    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    for repo_file in repo_files:
        functions.extend(
            get_functions_from_file(repository, repo_file, commit))

    return functions


def get_module_from_file(repo_file: str):
    '''
    Return the module from the file:
    '''
    with open(repo_file) as file:
        repo_file_content = file.read()
        repo_module = ast.parse(repo_file_content)
        return repo_module


def get_functions_from_file(repository: Repository,
                            repo_file: str,
                            diffs: DiffList=[],
                            commit: Commit=None):
    '''
    Returns the functions from the file.  They're created in the database if
    do not already exist
    '''
    functions = []

    repo_module = get_module_from_file(repo_file)
    for node in ast.walk(repo_module):
        if isinstance(node, ast.FunctionDef):
            relative_file_path = os.path.relpath(repo_file, repository.path)
            logger.info('Currently have node "{}" at path "{}@{}"'
                        .format(node.name, relative_file_path, node.lineno))
            function = get_function(
                repository=repository,
                node=node,
                file_path=relative_file_path,
                diffs=diffs,
                commit=commit,
            )
            logger.info('Got function {}'.format(function))

            functions.append(function)

    return functions


def get_function(repository: Repository,
                 node,
                 file_path: str,
                 commit: Commit=None,
                 diffs: DiffList=[],
                 ):
    '''
    Given a function's ast node information, return a corresponding Function.
    If it already exists in the database, use that one.  Otherwise make a new
    instance but do not save it in the DB. We make sure the function in the
    database and the AST node are the same by tracking line numbers in the
    database and looking at the diff.

    For example:
        - If the file was not editted and the node's line number is the same
          as the Function's latest FunctionHistory line number, then we have
          found the corresponding Function instance for the node
        - If the file was editted, we need to see how many lines were added or
          subtracted above the Function's latest FunctionHistory line number.

        - Since we just retrieved the node from the source code, the node has
          each Function's current line number, which we will store as the
          FunctionHistory line_number

    @param repository: the Repository instance we are getting a function from
    @node: the corresponding AST Function node
    @file_path: the relative file path of the function
    @commit (optional): the commit this function is a part of.  Necessary for
                        if we need the diff
    '''
    session = Session.object_session(repository)

    potential_matching_functions = get_all(
        session,
        Function,
        repository=repository,
        name=node.name,
        file_path=file_path)

    # we have found a possible matching function.  We now need to make sure
    # that we have the correct matching function by comparing the line numbers
    if potential_matching_functions:
        if not diffs:
            diffs = create_diffs(repository, commit)

        # We have at least one matching function, now we make sure it's the
        # exact corresponding function but making sure the lines are the same.

        # First, we check to see if there are any diffs in the file that would
        # influence the location of this function. If there are not, then the
        # AST Function node should have the exact same line number as the
        # matching function's latest FunctionHistory.line_number.
        affecting_diffs = [
            diff for diff in diffs if file_path == diff.file_path and
            diff.first_line <= node.lineno
        ]

        diff_impact = 0
        if affecting_diffs:
            # The file was altered, so we now need to know how many lines were
            # added or subtracted above the function.
            diff_impact = sum([diff.size_difference for diff in affecting_diffs])
            logger.debug('The following diffs are adding "{}" lines: {}'
                         .format(diff_impact, diffs))

        for function in potential_matching_functions:
            # if the function does not have node history, then that means it
            # is a new function that has not gone through the snapshot process
            if not function.function_history:
                continue

            if node.lineno == function.latest_history.first_line + diff_impact:
                function.node = node
                logger.debug('We found the corresponding Function instance: {}'
                             .format(function))
                return function

        logger.info('The following potential_matching_functions did not match  '
                    'the node line number "{lineno}": {functions}'
                    .format(lineno=node.lineno,
                            functions=potential_matching_functions))

    # There are no perfectly matching fucntions so we therefore will return
    # a new Function instance. We do not store the function at this point
    new_function = create(
        session,
        Function,
        repository=repository,
        node=node,
        file_path=file_path)

    logger.info('There is a new function: {}'.format(new_function))
    return new_function


def sync_mirror_repo(repository: Repository):
    '''
    Updates the mirror repository to match the code base the developer is
    working on
    '''
    # skip the .git directory, otherwise you are overwritting the commits in
    # the mirror repository
    if not os.path.exists(MIRROR_ROOT):
        os.makedirs(MIRROR_ROOT)

    if not os.path.exists(repository.mirror_path):
        logger.info('Initializing mirror repository')
        clone_repository(repository, repository.mirror_path)

    command = ('rsync -a {source} {destination} --exclude ".git"'
               .format(source=repository.original_path,
                       destination=MIRROR_ROOT))

    run_cmd(repository, command, log=True)


def get_patches_from_diffs(repository: Repository,
                           commit: Commit=None,
                           split_per_method=True) -> List[str]:
    '''
    Creates a patch file containing all the diffs in the repository and then
    returns all those patches as a list of patches
    '''
    import pdb; pdb.set_trace()

    # this command will output the diff information into stdout
    command = 'git --no-pager diff'
    if commit:
        command += ' {hash}~ {hash}'.format(hash=commit.commit_id)
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


def get_function_from_patch(repository: Repository,
                            patch: str,
                            file_path: str,
                            first_line: int,
                            last_line: int):
    '''
    Given a patch, find the corresponding function if possible
    '''
    session = Session.object_session(repository)
    pattern = re.compile('def \w*\(')
    matching_functions = pattern.findall(patch)
    if len(matching_functions) != 1:
        import pdb; pdb.set_trace()
        print('multiple matching_functions: ', matching_functions)

    else:
        function_name = matching_functions[0]

        # it comes out from the regex as def xxxxxx( so we need to splice out
        # just the function name
        function_name = function_name[4: -1]
        return get(
            session,
            Function,
            name=function_name,
            file_path=file_path,
        )


def create_diffs(repository: Repository,
                 commit: Commit=None,
                 is_synthetic=False,
                 function: Function=None) -> DiffList:
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
        file_path = patch.header.new_path

        # this only works for addition diffs
        first_line = 0
        for old_line_number, new_line_number, line in patch.changes:
            if not old_line_number and new_line_number:
                first_line = new_line_number
                break

        last_line = first_line + 1

        # TODO - make sure it can get the function from the patch
        if not function:
            function = get_function_from_patch(
                repository,
                patch.text,
                file_path,
                first_line,
                last_line)

        diff = create(
            session,
            Diff,
            commit=commit,
            first_line=first_line,
            last_line=last_line,
            patch=patch.text,
            function=function,
            file_path=file_path,
            is_synthetic=is_synthetic)

        diffs.append(diff)

    return diffs


def add_diff(diff: Diff):
    '''
    Adds the diff's contents to the source code
    '''
    _apply_diff(diff, revert=False)


def revert_diff(diff: Diff):
    '''
    Reverts the diff from the source
    '''
    _apply_diff(diff, revert=True)


def _apply_diff(diff: Diff, revert=False):
    '''
    Either reverts or applies a diff
    '''
    temp_output = None
    try:
        file_name = 'bugbuddy_diff_id_{}'.format(diff.id)
        suffix = '.patch'
        temp_output = tempfile.NamedTemporaryFile(
            prefix=file_name,
            suffix=suffix,
            dir=diff.commit.repository.path)
        temp_output.write(str.encode(diff.patch + '\n\n'))
        temp_output.flush()

        command = ('git apply {revert}{file_path}'
                   .format(revert='-R ' if revert else '',
                           file_path=temp_output.name))

        stdout, stderr = run_cmd(diff.commit.repository, command)
        if stderr:
            msg = ('Error trying to {revert_or_add} diff {diff} with patch:\n{patch}\n\n'
                   'stderr: {stderr}'
                   .format(revert_or_add='revert' if revert else 'add',
                           diff=diff,
                           patch=diff.patch,
                           stderr=stderr))
            raise BugBuddyError(msg)
    except Exception as e:
        logger.error('Hit error trying to apply diff: {}'.format(e))
    finally:
        if temp_output:
            temp_output.close()
