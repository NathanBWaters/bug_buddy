#!/usr/bin/env python3
'''
The Commit model.  A record of a particular change in a repository's code
'''

from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.errors import BugBuddyError
from bug_buddy.constants import TEST_OUTPUT_FAILURE, SYNTHETIC_CHANGE
from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Commit(Base):
    '''
    Schema representation of a change in the code repository.
    '''
    __tablename__ = 'commit'
    id = Column(Integer, primary_key=True)
    commit_id = Column(String, nullable=False)
    branch = Column(String, nullable=False)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='commits')

    # commit_type is an extra metadata around the content of the commit:
    #   1) DEVELOP - a standard change by a developer
    #   2) SYNTHETIC - a synthetic alteration
    #   3) RESET - a commit that resets synthetic alteration commits
    commit_type = Column(String, nullable=False)

    # For synthetic commits, we have a base commit that we break down into
    # smaller commits to determine blame.  The parent commit is the base commit
    # and all smaller commits the revert the diffs of the parent commit have
    # this attribute set.
    # parent_commit_id = Column(Integer, ForeignKey('commit.id'), nullable=True)
    # parent_commit = relationship(
    #     'Commit',
    #     remote_side=[parent_commit_id])
    # child_commits = relationship(
    #     'Commit',
    #     back_populates='parent_commit')

    # this is kind of a hack.  Synthetic commits are created from synthetic
    # diffs.  Retrieving the powerset of a set of diffs from the database is
    # prohibitively expensive.  So we will hash the sorted list of synthetic
    # diff ids and use that hash to retrieve the 'child' commits of a commit.
    synthetic_diff_hash = Column(Integer, nullable=True)

    test_runs = relationship('TestRun',
                             back_populates='commit',
                             cascade='all, delete, delete-orphan')

    # the corresponding functions histories created in this commit
    function_histories = relationship(
        'FunctionHistory',
        back_populates='commit',
        cascade='all, delete, delete-orphan')

    # the corresponding diffs created in this commit
    diff_links = relationship(
        'DiffCommitLink',
        back_populates='commit',
        cascade='all, delete, delete-orphan')

    def __init__(self,
                 repository: Repository,
                 commit_id: str,
                 branch: str,
                 commit_type: str=SYNTHETIC_CHANGE,
                 # parent_commit=None
                 ):  # Commit
        '''
        Creates a new TestResults instance
        '''
        if not commit_id:
            msg = 'Tried creating commit without a commit_id'
            raise BugBuddyError(msg)

        self.repository = repository
        self.commit_id = commit_id
        self.branch = branch
        self.commit_type = commit_type
        # self.parent_commit = parent_commit

    def causes_test_failures(self):
        '''
        Returns a bool if the commit causes any test failures
        '''
        for test_run in self.test_runs:
            for test_result in test_run.test_results:
                if test_result.status == TEST_OUTPUT_FAILURE:
                    return True

        return False

    @property
    def diffs(self):
        '''
        Returns the diffs in the commit
        '''
        return [diff_link.diff for diff_link in self.diff_links]

    @property
    def blames(self):
        '''
        Returns the diffs in the commit
        '''
        blames = []
        for test_run in self.test_runs:
            for test_result in test_run.test_results:
                blames.extend(test_result.blames)

        return blames

    def needs_blaming(self):
        '''
        Whether the commit needs to undergo the blame process
        '''
        return self.causes_test_failures() and not self.blames

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Commit {commit_id} | {branch} | {commit_type} />'
                .format(commit_id=self.commit_id,
                        branch=self.branch,
                        commit_type=self.commit_type))
