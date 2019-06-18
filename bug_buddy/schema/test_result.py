#!/usr/bin/env python3
'''
The TestResult model.  The pass/fail for a test at a particular commit.
'''
import numpy
from sqlalchemy import Column, ForeignKey, Integer, String, Float, Binary
from sqlalchemy.orm import relationship, deferred

from bug_buddy.constants import TEST_OUTPUT_FAILURE, TEST_OUTPUT_SUCCESS
from bug_buddy.schema.base import Base
from bug_buddy.errors import BugBuddyError
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

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='test_results')

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

    # this stores the tensor for determining which functions are to be blamed
    # for each test result. We used the 'deferred' function so that it's loaded
    # lazily and not always brought into memory.  For more information, read:
    #   https://docs.sqlalchemy.org/en/13/orm/loading_columns.html
    _cached_function_blame_feature = deferred(
        Column('cached_function_blame_feature', Binary, nullable=True))
    _cached_function_blame_label = deferred(
        Column('cached_function_blame_label', Binary, nullable=True))
    _cached_function_blame_feature_numpy = None
    _cached_function_blame_label_numpy = None

    blamed_function_prediction_vector = None

    @property
    def failed(self):
        '''
        Whether the test failed or not
        '''
        return self.status == TEST_OUTPUT_FAILURE

    @property
    def passed(self):
        '''
        Whether the test passed or not
        '''
        return self.status == TEST_OUTPUT_SUCCESS

    def __init__(self, test: Test, test_run: TestRun, status: str, time: float):
        '''
        Creates a new TestResults instance
        '''
        self.test = test
        self.test_run = test_run
        self.status = status
        self.time = time
        self.repository = test.repository

    @property
    def blamed_function_prediction(self):
        '''
        Returns test result prediction data
        '''
        if self.blamed_function_prediction_vector is None:
            msg = 'Requested prediction data but it does not exist'
            raise BugBuddyError(msg)

        prediction_dict = dict(zip(self.test_run.commit.functions,
                                   self.blamed_function_prediction_vector))

        return sorted(prediction_dict.items(), key=lambda x: x[1], reverse=True)[: 5]

    @property
    def cached_function_blame_feature(self):
        '''
        Returns the test run feature as a 1D numpy array
        '''
        if self._cached_function_blame_feature_numpy is None:
            if self._cached_function_blame_feature:
                self._cached_function_blame_feature_numpy = numpy.fromstring(
                    self._cached_function_blame_feature)
            else:
                return None

        return numpy.ravel(self._cached_function_blame_feature_numpy)

    @property
    def cached_function_blame_label(self):
        '''
        Returns the test run label as a 1D numpy array
        '''
        if self._cached_function_blame_label_numpy is None:
            if self._cached_function_blame_label:
                self._cached_function_blame_label_numpy = numpy.fromstring(
                    self._cached_function_blame_label)
            else:
                return None

        return numpy.ravel(self._cached_function_blame_label_numpy)

    def summary(self, indent=0):
        '''
        Prints a summary to terminal about this test run
        '''
        print(' ' * indent + str(self))
        for blame in self.blames:
            print((' ' * (indent + 2)) + str(blame))

    def get_blamed_function(self, indent=0):
        '''
        Prints a summary to terminal about this test run
        '''
        print(' ' * indent + str(self))
        for blame in self.blames:
            print((' ' * (indent + 2)) + str(blame))

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
