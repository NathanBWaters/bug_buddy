'''
Given a repository and commit, it will store the change data into the database.
'''
from collections import defaultdict
import difflib
import os
import random
import traceback
from typing import List
from sqlalchemy.orm.exc import NoResultFound
import sys

from bug_buddy.constants import PYTHON_FILE_TYPE, DEVELOPER_CHANGE
from bug_buddy.db import create, Session, session_manager
from bug_buddy.errors import UserError, BugBuddyError
from bug_buddy.git_utils import (
    create_commit,
    git_push,
    get_previous_commit,
    get_most_recent_commit,
    revert_commit,
    is_repo_clean)
from bug_buddy.logger import logger
from bug_buddy.source import (
    get_diff_patches,
    get_range_of_patch,
    get_function_nodes_from_repo,
    get_function_from_patch)
from bug_buddy.schema import (
    Commit,
    Diff,
    Function,
    FunctionHistory,
    FunctionToTestLink,
    Repository,
    TestRun)
from bug_buddy.schema.aliases import FunctionList, DiffList


PREVIOUS_HISTORY = 'PREVIOUS_HISTORY'
CURRENT_NODE = 'CURRENT_NODE'


def snapshot(repository: Repository, allow_empty=False):
    '''
    Snapshots a dirty commit tree and records everything
    '''
    commit = None
    try:
        session = Session.object_session(repository)

        commit = create_commit(repository,
                               commit_type=DEVELOPER_CHANGE,
                               allow_empty=allow_empty)

        snapshot_commit(repository, commit)

        git_push(repository)

        session.commit()
        return commit

    except Exception as e:
        traceback.print_exc()
        import pdb; pdb.set_trace()
        # revert all the local edits
        logger.error('Hit the following exception: {}'.format(e))

        if commit:
            logger.error('Reverting local changes')
            revert_commit(repository, commit)

        raise e


def snapshot_commit(repository: Repository, commit: Commit):
    '''
    Given a repository and commit, store the necessary data such as the
    Functions, FunctionHistory, and Diff instances.

    It will do this in the following order:
        - Get the diffs
        - Get the functions
        - Store new functions
        - Create new FunctionHistory for all functions that exist
        - Store the diffs with their corresponding FunctionHistory
    '''
    logger.info('Snapshotting commit {}'.format(commit))

    # retrieve the AST function nodes from the repository
    function_nodes = get_function_nodes_from_repo(repository)

    # retrieve the patches from the repository in the form of whatthepatch
    # patch objects
    patches = get_diff_patches(commit)

    # create FunctionHistory instances for each Function
    save_function_histories(repository, commit, function_nodes, patches)

    # create Diff instances
    diffs = create_diffs(repository, commit)

    # save the diffs
    save_diffs(repository, commit, diffs)


def save_function_histories(repository: Repository,
                            commit: Commit,
                            function_nodes,
                            patches):
    '''
    Stores the function histories
    '''
    session = Session.object_session(repository)

    previous_commit = get_previous_commit(commit)

    if not previous_commit:
        logger.info('No previous commit, creating new function nodes')
        create_new_functions_from_nodes(commit, function_nodes)
        return

    # from the patches, determine which files were altered
    altered_files = list(set([patch.header.new_path for patch in patches]))
    logger.info('Altered files: {}'.format(altered_files))

    altered_function_nodes = []
    unaltered_function_nodes = []
    for function_node in function_nodes:
        if function_node.file_path in altered_files:
            altered_function_nodes.append(function_node)
        else:
            unaltered_function_nodes.append(function_node)

    # If the file was not altered, then we can simply find the previous
    # function history and recreate it for this commit without even finding
    # the appropriate function
    for function_node in unaltered_function_nodes:
        try:
            previous_function_history = (
                session.query(FunctionHistory)
                       .join(FunctionHistory.function)
                       .filter(FunctionHistory.commit_id == previous_commit.id)
                       .filter(Function.name == function_node.name)
                       .filter(Function.file_path == function_node.file_path)
                       .filter(FunctionHistory.first_line == function_node.lineno)
            ).one()
        except NoResultFound:
            import pdb; pdb.set_trace()
            logger.error('Unable to find previous function history for node {}'
                         'which was in an unaltered file')

        function_history = create(
            session,
            FunctionHistory,
            function=previous_function_history.function,
            commit=commit,
            node=function_node,
            first_line=previous_function_history.first_line,
            last_line=previous_function_history.last_line)

        logger.info('Created unaltered function history: {}'
                    .format(function_history))

    # If the file was altered, then we need to be extremely careful about how
    # we track function history.
    _save_altered_file_function_history(
        commit, previous_commit, altered_files, altered_function_nodes, patches)


def _save_altered_file_function_history(commit: Commit,
                                        previous_commit: Commit,
                                        altered_file_paths: List[str],
                                        altered_function_nodes,
                                        patches):
    '''
    Saves the function history for altered files

    If the file was altered, then we need to be extremely careful about how
    we track function history.  Currently follows the following algorithm:
     1) Create a dictionary where the key is the function name of the values
        is a dictionary which has two keys: PREVIOUS_HISTORY and CURRENT_NODE.
        They map to a list of the matching function histories and nodes.
     2) If there are the same number of old function and new node, then they are
        matched and removed from the dictionary.
     3) If there is more than one function/new node pairing for a single key,
        then they will be matched in order of first_line.
     4) Remaining nodes will look into the patches to see if they were renamed.
        Those are then removed from the dictionary.
     5) Remaining nodes are treated as newly created functions.
     6) Remaining old functions are treated as deleted nodes.
    '''
    session = Session.object_session(commit)

    for altered_file in altered_file_paths:
        # get the function nodes present in the specified file
        file_current_function_nodes = [
            function_node for function_node in altered_function_nodes
            if function_node.file_path == altered_file]

        # get the previous commit's version of the file's function histories
        # in order by their first line
        file_previous_function_histories = [
            function_history for function_history
            in previous_commit.function_histories
            if function_history.function.file_path == altered_file]

        # combine the file's current nodes and previous histories into a
        # dictionary with key being their names
        function_name_map = {}
        for func in (file_current_function_nodes + file_previous_function_histories):
            func_type = (PREVIOUS_HISTORY if isinstance(func, FunctionHistory)
                         else CURRENT_NODE)
            if function_name_map.get(func.name):
                function_name_map[func.name][func_type].append(func)
            else:
                function_name_map[func.name] = defaultdict(list)
                function_name_map[func.name][func_type].append(func)

        for func_name, corresponding_functions in function_name_map.items():
            previous_histories = corresponding_functions[PREVIOUS_HISTORY]
            current_nodes = corresponding_functions[CURRENT_NODE]

            matched_pairs, unmatched_nodes = _match_nodes_with_history(
                previous_histories, current_nodes)

            for node, previous_history in matched_pairs:
                function_history = create(
                    session,
                    FunctionHistory,
                    function=previous_history.function,
                    commit=commit,
                    node=node,
                    first_line=node.first_line,
                    last_line=node.last_line)

                logger.info('Created altered function history: {}'
                            .format(function_history))

            # convert all unmatched nodes into new functions
            create_new_functions_from_nodes(commit, unmatched_nodes)


def create_new_functions_from_nodes(commit: Commit, function_nodes):
    '''
    Given a list of function nodes, it will create new functions
    '''
    session = Session.object_session(commit)
    for node in function_nodes:
        # create the function instance
        function = create(
            session,
            Function,
            repository=commit.repository,
            node=node,
            file_path=node.file_path)

        # We have a new function!
        function_history = create(
            session,
            FunctionHistory,
            function=function,
            commit=commit,
            node=node,
            first_line=node.first_line,
            last_line=node.last_line)

        logger.info('Created new function history: {}'.format(function_history))


def _match_nodes_with_history(previous_histories, current_nodes):
    '''
    Given a list of previous histories and a list of current_nodes that all have
    the same function name, try matching based on the difference in their
    content.  The more similar they are, the more likely that they are the same
    function
    '''
    matched_pairs = []

    # if either list of items becomes empty, then we have completed
    while previous_histories and current_nodes:
        highest_match_ratio = 0
        match = None

        for node in current_nodes:
            for history in previous_histories:
                # Find the difference between the given node and history
                matcher = difflib.SequenceMatcher(
                    a=node.source_code, b=history.source_code)
                if matcher.ratio() > highest_match_ratio:
                    match = (node, history)

        current_nodes.remove(match[0])
        previous_histories.remove(match[1])

        # store the match
        matched_pairs.append(match)

    return matched_pairs, current_nodes


def save_diffs(repository: Repository,
               commit: Commit,
               diffs: DiffList):
    '''
    Saves the diffs
    '''
    session = Session.object_session(repository)
    for diff in diffs:
        diff.commit = commit

    session.add_all(diffs)


def create_diffs(repository: Repository,
                 commit: Commit=None,
                 is_synthetic=False,
                 function: Function=None,
                 allow_empty=True) -> DiffList:
    '''
    Returns a list of diffs from a repository
    '''
    session = Session.object_session(repository)
    if not commit:
        commit = get_most_recent_commit(repository)

    diffs = []

    # the patches should be split on a per function basis
    patches = get_diff_patches(commit)

    if not allow_empty and not patches:
        import pdb; pdb.set_trace()
        logger.error('No diffs discovered when allow_no_diffs == False')
        get_diff_patches(commit)

    for patch in patches:
        diff_function = function
        file_path = patch.header.new_path
        first_line, last_line = get_range_of_patch(patch)

        if not diff_function:
            function_history = commit.get_corresponding_function(
                file_path=file_path,
                start_range=first_line,
                end_range=last_line,
            )
            diff_function = function_history.function if function_history else None

        diff = create(
            session,
            Diff,
            commit=commit,
            patch=patch.text,
            function=diff_function,
            file_path=file_path,
            is_synthetic=is_synthetic,
            first_line=first_line,
            last_line=last_line)

        diffs.append(diff)

    return diffs
