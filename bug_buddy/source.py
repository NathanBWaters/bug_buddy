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
from bug_buddy.schema.aliases import FunctionList, DiffList, FunctionHistoryList


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
        Is called when the transformer hits a function node.  Note that child
        nodes of nodes that have a custom visitor method won’t be visited
        unless the visitor calls generic_visit() or visits them itself, so we
        need to do that.
        '''
        # Add extra attributes to the node
        node.file_path = os.path.relpath(self.file_path, self.repository.path)
        node.first_line = node.lineno
        node.last_line = node.body[-1].lineno
        node.source_code = astor.to_source(node)

        # store the function that was visited
        self.function_nodes.append(node)

        for child_node in node.body:
            self.generic_visit(child_node)


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

    command = ('rsync -a {source}/ {destination} --exclude ".git"'
               .format(source=repository.original_path,
                       destination=repository.mirror_path))

    run_cmd(repository, command, log=False)

    # remove the pyc files so that we do not run into any import errors when
    # trying to run the testing commands:
    # http://wisercoder.com/importmismatcherror-python-fix/
    clean_command = 'find . -name \*.pyc -delete'
    run_cmd(repository, clean_command, log=False)


def get_diff_patches(commit: Commit=None,
                     split_per_method=True,
                     only_unstaged=False) -> List[str]:
    '''
    Creates a patch file containing all the diffs in the repository and then
    returns all those patches as a list of patches
    '''
    # this command will output the diff information into stdout
    command = 'git --no-pager diff'
    if commit:
        command += ' {hash}~ {hash}'.format(hash=commit.commit_id)
    diff_data, _ = run_cmd(commit.repository, command)

    # TODO: MY GOODNESS this is such a hack
    if not diff_data:
        command = 'git --no-pager diff'
        diff_data, _ = run_cmd(commit.repository, command)

    # TODO: GETTING EVEN WORSE.  This is when there is a commit already created
    if not diff_data and not only_unstaged:
        command = 'git diff origin/bug_buddy..HEAD'
        diff_data, _ = run_cmd(commit.repository, command)

    raw_patches = diff_data.split('diff --git ')[1:]

    # covert the list of patches into whatthepatch patch objects
    patches = [list(whatthepatch.parse_patch(patch))[0]
               for patch in raw_patches]

    # if split_per_method and commit.function_histories:
    #     patches = _split_patches_by_method(commit, patches)

    return patches


def _split_patches_by_method(commit: Commit, patches):
    '''
    Given whatthepatch patches, it will split them up by methods so one patch
    doesn't go across multiple methods
    '''
    granular_patches = []

    for patch in patches:
        granular_patches.extend(_split_patch_by_method(commit, patch))

    return granular_patches


def _split_patch_by_method(commit: Commit, patch):
    '''
    Given whatthepatch patches, it will split them up by methods so one patch
    doesn't go across multiple methods
    '''
    granular_patches = []
    start_range, end_range = get_range_of_patch(patch)

    histories = commit.get_function_histories(
        file_path=patch.header.new_path,
        start_range=start_range,
        end_range=end_range,
    )

    if len(histories) > 1:
        sub_patches = []
        # this patch spans across multiple functions, we need to split it
        # up.  We are going to split the raw text at the start point of
        # each function.
        for history in histories:
            sub_patch = _match_patch_with_history(patch, history)

            if not sub_patch.changes:
                # If the parsed patch does not have any changes, then
                # nothing happened to that function and we can remove this
                # part of the patch
                msg = (
                    'No changes for the parsed_patch for sub_patch:\n'
                    '{sub_patch}\nThis was range {start}-{end} of:\n'
                    '{patch}'
                    .format(sub_patch=sub_patch,
                            start=start,
                            end=end,
                            patch=patch))
                logger.debug(msg)
                import pdb; pdb.set_trace()

            sub_patches.append(sub_patch)

            granular_patches.extend(sub_patches)

        else:
            granular_patches.append(patch)

    return granular_patches


def _match_patch_with_history(patch, function_histories: FunctionHistoryList):
    '''
    Returns a modified form of the patch for the function history

    It would turn following patch:

        def a:
            dog = 'dog'
    +        # added to a

            def b:
                cat = 'cat'
    +            # added to b

            more_dog_stuff = 2
    +        # added more to a


    Into two patches:
    1) for function history 'def a'

        def a:
            dog = 'dog'
    +        # added to a

            def b:
                cat = 'cat'

            more_dog_stuff = 2
    +        # added more to a

    2) for function history 'def b''

        def a:
            dog = 'dog'

            def b:
                cat = 'cat'
    +            # added to b

            more_dog_stuff = 2
    '''
    for function_history in function_histories:
        function_lines = list(range(function_history.first_line,
                                    function_history.last_line))

        # now remove the lines that are a part of a function that is within
        # this function.  This an inner function and changes to that function
        # should not relate to the patch for this function.
        for other_history in function_histories:
            if (other_history.first_line > function_history.first_line and
                    other_history.last_line < function_history.last_line):
                logger.info('{} is an inner function of {}'
                            .format(other_history, function_history))

                inner_func_lines = list(range(other_history.first_line,
                                              other_history.last_line))

                # remove the inner function lines from the list of lines in the
                # patch that maps to this function
                function_lines = [line for line in function_lines
                                  if line not in inner_func_lines]

        function_changes = []
        for original_line, new_line, change in patch.changes:
            if new_line in function_lines:
                pass

        # for change in pat


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

    logger.info('{}-{}'.format(start_range, end_range))
    # if end_range - start_range > 25 or (start_range == 0 and end_range == 132):
    #     import pdb; pdb.set_trace()
    #     logger.error('What in tarnation')

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
        import pdb; pdb.set_trace()
        logger.error('Hit error trying to apply diff: {}'.format(e))
    finally:
        if temp_output:
            temp_output.close()
