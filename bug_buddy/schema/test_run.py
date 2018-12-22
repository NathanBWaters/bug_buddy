#!/usr/bin/env python3
'''
The TestRun model.  Represents a run of multiple tests against a single commit.
This is important because you can have multiple TestRun instances against a
single commit
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import time

from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class TestRun(Base):
    '''
    Schema representation of a single test run, which could involve multiple
    tests where each has its own test result.
    '''
    __tablename__ = 'test_run'
    id = Column(Integer, primary_key=True)

    started_timestamp = Column(Integer, nullable=False)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='test_runs')

    output_file = Column(String(500), nullable=False)

    test_results = relationship('TestResult', back_populates='test_run')

    def __init__(self,
                 commit: Commit,
                 output_file: str,
                 started_timestamp: int=None):
        '''
        Creates a new TestResults instance
        '''
        if not started_timestamp:
            started_timestamp = time.time()

        self.started_timestamp = started_timestamp
        self.output_file = output_file
        self.commit = commit
