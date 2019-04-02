#!/usr/bin/env python3
'''
The TestResult model.  The pass/fail for a test at a particular commit.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import TEST_OUTPUT_FAILURE
from bug_buddy.schema.base import Base
from bug_buddy.schema.test import Test
from bug_buddy.schema.test_run import TestRun


class TestResult(Base):
    '''
    Schema representation of a test result.  Links a Test with a run, which
    further links to the code.  Stores whether or not the test passed/failed for
    that run
    '''
    __tablename__ = 'test_result'
    id = Column(Integer, primary_key=True)
    test_id = Column(Integer, ForeignKey('test.id'))
    test = relationship('Test', back_populates='test_results')

    test_run_id = Column(Integer, ForeignKey('test_run.id'))
    test_run = relationship('TestRun', back_populates='test_results')

    time = Column(Float, nullable=False)

    # Whether or not the test passed or failed
    status = Column(String, nullable=False)

    function_maps = relationship(
        'FunctionToTestLink',
        back_populates='test_result',
        cascade='all, delete, delete-orphan')

    # get all the blames for this TestResult
    blames = relationship(
        'Blame',
        back_populates='test_result',
        cascade='all, delete, delete-orphan'
    )

    # we need a unique constraint on test results
    @property
    def failed(self):
        '''
        Whether the test failed or not
        '''
        return self.status == TEST_OUTPUT_FAILURE

    def __init__(self, test: Test, test_run: TestRun, status: str, time: float):
        '''
        Creates a new TestResults instance
        '''
        self.test = test
        self.test_run = test_run
        self.status = status
        self.time = time

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<TestResult {id} | {file}.{classname}.{name} | {status} | '
                'test_id={test_id} />'
                .format(id=self.id,
                        name=self.test.name,
                        file=self.test.file,
                        classname=self.test.classname,
                        status=self.status,
                        test_id=self.test.id))
