'''
Assigns blame for test failures to a commit
'''
import itertools
from junitparser import JUnitXml

from bug_buddy.constants import (ERROR_STATEMENT,
                                 SYNTHETIC_FIXING_CHANGE)
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.git_utils import create_commit
from bug_buddy.logger import logger
from bug_buddy.runner import run_test
from bug_buddy.schema import (
    Repository,
    Blame,
    Diff,
    DiffCommitLink,
    TestResult,
    TestRun,
    Commit)
from bug_buddy.snapshot import snapshot_commit


def synthetic_blame(repository: Repository,
                    commit: Commit,
                    test_run: TestRun):
    '''
    Given a synthetic commit, it will create blames for the commit based on
    the blames of the sub-combinations of the diffs
    '''
    session = Session.object_session(repository)
    diffs = commit.diffs
    diff_ids = [diff.id for diff in diffs]
    test_failures = test_run.failed_tests

    # Get the list of commits that are made up of the subset of diffs in this
    # commit
    children_commit_ids = (
        session.query(DiffCommitLink).
        filter(DiffCommitLink.diff_id.in_(diff_ids)).
        group_by(DiffCommitLink.commit_id).all())
    children_commits = (
        session.query(Commit)
        .filter(Commit.id.in_(children_commit_ids)).all())

    for test_failure in test_failures:
        # get all the blames for this test failure that were new at the time.
        # The newness attribute should remove duplicates.
        # All of these blames will now be combined for a new blame for this
        # test failure.

        # Any diff that is in this commit that wasn't in the commits 

        # It is a completely new test_failure, that cannot be attributed to any
        # previous edits if:
        #  

        # Requirement for memoized blame:
        #  1) Get all 
        previous_failing_instances = test_failure.test.failing_instances
        failing_instances_ids = [failing_instance.id for failing_instance
                                 in previous_failing_instances]


        # session.query(Address).filter(Address.person == person).all()


def get_matching_commit_for_diffs(repository, diff_set):
    '''
    Given a set of diffs, return if there is a commit that has those diffs
    '''
    diff_ids = [diff.id for diff in diff_set]
    session = Session.object_session(repository)

    if not diff_ids:
        # return a commit that does not have a diff
        return session.query(Commit).filter(Commit.diff_links == None).first()

    # get the commit that has a mapping to those two diffs
    matching_commits = (
        session.query(Commit)
               .join(DiffCommitLink)
               .filter(DiffCommitLink.diff_id.in_(diff_ids)).all())
    for matching_commit in matching_commits:
        commit_diff_ids = [diff.id for diff in matching_commit.diffs]
        if sorted(diff_ids) == sorted(commit_diff_ids):
            return matching_commit


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

