#!/usr/bin/env python3
'''
The TestRun model.  Represents a run of multiple tests against a single commit.
This is important because you can have multiple TestRun instances against a
single commit
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class TestRun(Base):
    '''
    Schema representation of a test result.  Links a Test with a run, which
    further links to the code.  Stores whether or not the test failed for that
    run
    '''
    __tablename__ = 'test_run'
    id = Column(Integer, primary_key=True)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='test_runs')

    test_results = relationship('TestResult', back_populates='test_run')
    test_outputs = relationship('TestOutput', back_populates='test_run')

    # Whether or not the test passed or failed
    status = Column(String, nullable=False)

    def __init__(self, commit: Commit):
        '''
        Creates a new TestResults instance
        '''
        self.commit = commit
