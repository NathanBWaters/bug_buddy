'''
Assigns blame for test failures to a commit
'''
import itertools
from junitparser import JUnitXml

from bug_buddy.constants import (ERROR_STATEMENT,
                                 TEST_OUTPUT_FAILURE,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import (
    Session,
    get,
    get_or_create,
    create,
    session_manager,
    get_all)
from bug_buddy.git_utils import create_commit
from bug_buddy.logger import logger
from bug_buddy.runner import run_test
from bug_buddy.schema import (
    Repository,
    Blame,
    Diff,
    TestResult,
    TestRun,
    Commit)
from bug_buddy.schema.aliases import DiffList


def synthetic_blame_all_commits(repository: Repository):
    '''
    Synthetically blames all commits in order of the number of diffs that they
    have associated with them for a repository
    '''
    session = Session.object_session(repository)
    commits = get_all(session, Commit, repository_id=repository.id)
    commits.sort(key=lambda commit: len(commit.diffs), reverse=False)
    for commit in commits:
        # if the commit has already been blamed, no need to blame it again
        if commit.needs_blaming():
            test_run = commit.test_runs[0]
            synthetic_blame(test_run)


def synthetic_blame(commit: Commit,
                    test_run: TestRun):
    '''
    Given a synthetic commit, it will create blames for the commit based on
    the blames of the sub-combinations of the diffs
    '''
    if not test_run.failed_tests:
        logger.info('No failing tests for commit {}, nothing to blame'
                    .format(commit))
        return

    logger.info('Setting blame for commit: {}'.format(commit))
    session = Session.object_session(commit)

    # Get the list of commits that are made up of the subset of diffs in this
    # commit
    children_commits = []
    for diff_set in powerset(commit.diffs):
        # an empty set in the powerset is ignored
        if not diff_set:
            continue

        # there is only one diff_set that is the same length as the commit's
        # diff, which means they're the exact same diff set, so it's not really
        # a child of the commit
        if len(diff_set) == len(commit.diffs):
            break

        diff_hash = get_diff_set_hash(diff_set)
        child_commit = (
            session.query(Commit)
            .filter(Commit.synthetic_diff_hash == diff_hash).one())
        children_commits.append(child_commit)

    for test_failure in test_run.failed_tests:
        # get all the blames for this test failure that were new at the time.
        # The newness attribute should remove duplicates.
        # All of these blames will now be combined for a new blame for this
        # test failure.
        children_test_failure_blames = []
        for child_commit in children_commits:
            if child_commit.has_same_test_result_output(
                    test_failure,
                    status=TEST_OUTPUT_FAILURE):
                child_test_failure = (
                    child_commit.get_matching_test_result(test_failure))

                for blame in child_test_failure.blames:
                    children_test_failure_blames.append(blame)

        if children_test_failure_blames:
            faulty_diffs = list(set(
                [blame.diff for blame in children_test_failure_blames]))

            for faulty_diff in faulty_diffs:
                logger.info('Assigning blame using child commit {} blame {} '
                            'for test failure: {}'
                            .format(child_commit, faulty_diff, test_failure))
                create(session,
                       Blame,
                       diff=faulty_diff,
                       test_result=test_failure)

        else:
            # We have created a completely new blame from this combination of
            # diffs in comparison from its child

            for diff in commit.diffs:
                blame = create(
                    session,
                    Blame,
                    diff=diff,
                    test_result=test_failure)
                logger.info('Assigning new blame for commit {} blame {}'
                            'for test failure: {}'
                            .format(commit, blame, test_failure))

    logger.info('Completed blaming for {}'.format(commit))


def get_matching_commit_for_diffs(repository, diff_set):
    '''
    Given a set of diffs, return if there is a commit that has those diffs
    '''
    session = Session.object_session(repository)
    diff_hash = get_diff_set_hash(diff_set)
    return get(session, Commit, synthetic_diff_hash=diff_hash)


def get_diff_set_hash(diffs: DiffList):
    '''
    Given a list of diffs, return the hash
    '''
    base_ids = [diff.base_synthetic_diff_id for diff in diffs]
    base_ids.sort()
    return hash(frozenset(base_ids))


def powerset(diffs):
    '''
    Returns the powerset of the diffs except the empty set

    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"

    @param: list of diffs
    @returns: powerset of the diffs
    '''
    return (itertools.chain.from_iterable(
        itertools.combinations(diffs, index) for index in range(len(diffs) + 1)
    ))

