'''
Code for interacting with the source code
'''
import ast
import astor
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


class AstTreeWalker(ast.NodeTransformer):
    '''
    Used for walking the AST tree for a repository
    '''
    def __init__(self,
                 repository: Repository,
                 file_path: str,
                 commit: Commit=None,
                 prepend_assert_false=False):
        '''
        Creates a AstTreeWalker Ast Transformer
        '''
        self.repository = repository
        self.commit = commit
        self.prepend_assert_false = prepend_assert_false
        self.file_path = file_path
        self.num_edits = 0

        # stores teh FunctionDef ast nodes
        self.function_nodes = []

    def visit_FunctionDef(self, node):
        '''
        Is called when the transformer hits a function node
        '''
        # Add extra attributes to the node
        node.file_path = os.path.relpath(self.file_path, self.repository.path)
        node.first_line = node.lineno
        node.last_line = node.body[-1].lineno
        node.source_code = astor.to_source(node)

        # store the function that was visited
        self.function_nodes.append(node)


def get_function_nodes_from_repo(repository: Repository):
    '''
    Returns the AST FunctionDef nodes from the repository src files
    '''
    # collect all the files
    repo_files = repository.get_src_files(filter_file_type=PYTHON_FILE_TYPE)

    function_nodes = []
    for repo_file in repo_files:
        file_module = get_module_from_file(repo_file)
        transformer = AstTreeWalker(repository=repository,
                                    file_path=repo_file)
        transformer.visit(file_module)
        function_nodes.extend(transformer.function_nodes)

    return function_nodes


def get_module_from_file(repo_file: str):
    '''
    Return the module from the file:
    '''
    with open(repo_file) as file:
        repo_file_content = file.read()
        repo_module = ast.parse(repo_file_content)
        return repo_module


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


def get_diff_patches(commit: Commit=None,
                     split_per_method=True) -> List[str]:
    '''
    Creates a patch file containing all the diffs in the repository and then
    returns all those patches as a list of patches
    '''
    # this command will output the diff information into stdout
    command = 'git --no-pager diff'
    if commit:
        command += ' {hash}~ {hash}'.format(hash=commit.commit_id)
    diff_data, _ = run_cmd(commit.repository, command)

    # import pdb; pdb.set_trace()
    raw_patches = diff_data.split('diff --git ')[1:]

    # covert the list of patches into whatthepatch patch objects
    patches = [list(whatthepatch.parse_patch(patch))[0]
               for patch in raw_patches]

    if split_per_method and commit.function_histories:
        patches = _split_patches_by_method(commit, patches)

    return patches


def _split_patches_by_method(commit: Commit, patches):
    '''
    Given whatthepatch patches, it will split them up by methods so one patch
    doesn't go across multiple methods
    '''
    granular_patches = []

    for patch in patches:
        start_range, end_range = get_range_of_patch(patch)
        histories = commit.get_function_histories(
            file_path=patch.header.new_path,
            start_range=start_range,
            end_range=end_range,
        )

        if len(histories) > 1:
            # this patch spans across multiple functions, we need to split it
            # up.  We are going to split the raw text at the start point of
            # each function.
            split_ranges = []
            raw_patch_lines = patch.text.split('\n')
            header = raw_patch_lines[0:4]

            for i in range(len(histories)):
                if i == 0:
                    split_ranges.append((0, histories[i + 1].first_line))

                elif i == len(histories) - 1:
                    split_ranges.append(
                        (histories[i].first_line, len(raw_patch_lines) + 1))

                else:
                    split_ranges.append(
                        (histories[i].first_line, histories[i + 1].first_line))

            sub_patches = []
            for start, end in split_ranges:
                sub_patch = '\n'.join(
                    header + raw_patch_lines[start: end])
                sub_patches.append(sub_patch)

            granular_patches.extend(sub_patches)

        else:
            granular_patches.append(patch)

    return granular_patches


def get_range_of_patch(patch):
    '''
    Given a whatthepatch patch, it will return the start and end range of the
    patch.  We consider the start of the patch not to be the first line
    necessarily, but the first line that is changed.
    '''
    start_range = None
    end_range = None
    for original_line, new_line, change in patch.changes:
        if original_line is None or new_line is None:
            if not start_range:
                start_range = new_line or original_line

            if not end_range or (new_line or -1) > end_range:
                end_range = new_line or original_line

    if start_range is None or end_range is None:
        import pdb; pdb.set_trace()
        logger.error('Failed to get start_range or end_range')

    return start_range, end_range


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
        # import pdb; pdb.set_trace()
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


def apply_diff(diff: Diff):
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
