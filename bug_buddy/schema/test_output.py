#!/usr/bin/env python3
'''
The TestResult model.  The pass/fail for a test at a particular commit.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.test import Test
from bug_buddy.schema.test_run import TestRun


class TestOutput(Base):
    '''
    Schema representation of a run.  A run links test runs with a snapshot of
    the code at a point in time.
    '''
    __tablename__ = 'test_output'
    # Here we define columns for the table address.
    # Notice that each column is also a normal Python instance attribute.
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey('test.id'))
    test = relationship('Test', back_populates='test_outputs')

    test_run_id = Column(Integer, ForeignKey('test_run.id'))
    test_run = relationship('TestRun', back_populates='test_outputs')

    # Whether the test ran or not
    results = Column(String, nullable=False)

    # whether or not the TestRun is based on a commit that was created
    # synthetically
    is_synthetic = Column(Boolean, nullable=False)

    commit = Column(String(250))

    def __init__(self, test_run: TestRun, test: Test, is_synthetic: bool, result: str):
        '''
        Creates a new TestResults instance
        '''
        self.test_run = test_run
        self.test = test
        self.result = result
        self.is_synthetic = is_synthetic
