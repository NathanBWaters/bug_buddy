#!/usr/bin/env python3
'''
The Commit model.  A record of a particular change in a repository's code
'''

from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import FAILURE, SYNTHETIC_CHANGE
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

    test_runs = relationship('TestRun',
                             back_populates='commit',
                             cascade='all, delete, delete-orphan')

    def __init__(self,
                 repository: Repository,
                 commit_id: str,
                 branch: str,
                 commit_type: str=SYNTHETIC_CHANGE):
        '''
        Creates a new TestResults instance
        '''
        self.repository = repository
        self.commit_id = commit_id
        self.branch = branch
        self.commit_type = commit_type

    def causes_test_failures(self):
        '''
        Returns a bool if the commit causes any test failures
        '''
        for test_run in self.test_runs:
            for test_result in test_run.test_results:
                if test_result.status == FAILURE:
                    return True

        return False

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Commit {commit_id} | {branch} | {commit_type} />'
                .format(commit_id=self.commit_id,
                        branch=self.branch,
                        commit_type=self.commit_type))
