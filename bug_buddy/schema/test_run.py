#!/usr/bin/env python3
'''
The TestRun model.  Represents a run of multiple tests against a single commit.
This is important because you can have multiple TestRun instances against a
single commit
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import textwrap
import time

from bug_buddy.constants import (TEST_OUTPUT_FAILURE,
                                 TEST_OUTPUT_SKIPPED,
                                 TEST_OUTPUT_SUCCESS)
from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class TestRun(Base):
    '''
    Schema representation of a single test run, which could involve multiple
    tests where each has its own test result.
    '''
    __tablename__ = 'test_run'
    id = Column(Integer, primary_key=True)

    start_timestamp = Column(Integer, nullable=False)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='test_runs')

    test_results = relationship('TestResult',
                                back_populates='test_run',
                                cascade='all, delete, delete-orphan')

    def __init__(self,
                 commit: Commit,
                 start_timestamp: int=None):
        '''
        Creates a new TestResults instance
        '''
        if not start_timestamp:
            start_timestamp = time.time()

        self.start_timestamp = start_timestamp
        self.commit = commit

    @property
    def num_passed_tests(self):
        '''
        Returns the number of passing tests in this test run
        '''
        return len(self.passed_tests)

    @property
    def passed_tests(self):
        '''
        Returns the list of passing tests in this test run
        '''
        return [test for test in self.test_results
                if test.status == TEST_OUTPUT_SUCCESS]

    @property
    def num_failed_tests(self):
        '''
        Returns the number of failed tests in this test run
        '''
        return len(self.failed_tests)

    @property
    def failed_tests(self):
        '''
        Returns the list of failed tests in this test run
        '''
        return [test for test in self.test_results
                if test.status == TEST_OUTPUT_FAILURE]

    @property
    def num_skipped_tests(self):
        '''
        Returns the number of skipped tests in this test run
        '''
        return len(self.skipped_tests)

    @property
    def skipped_tests(self):
        '''
        Returns the list of skipped tests in this test run
        '''
        return [test for test in self.test_results
                if test.status == TEST_OUTPUT_SKIPPED]

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        passed = self.num_passed_tests
        failed = self.num_failed_tests
        skipped = self.num_skipped_tests
        total = passed + failed + skipped
        return ('<TestRun {id} | {commit_id} | Passed: {passed} | '
                'Failed: {failed} | Skipped: {skipped} | total={total} />'
                .format(id=self.id,
                        commit_id=self.commit_id,
                        passed=passed,
                        failed=failed,
                        skipped=skipped,
                        total=total))
