#!/usr/bin/env python3
'''
Contains data between one method and one test for a single commit.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.function_history import FunctionHistory
from bug_buddy.schema.test_result import TestResult


class BlameData(Base):
    '''
    Contains data between one method and one test for a single commit.
    '''
    __tablename__ = 'blame_data'
    id = Column(Integer, primary_key=True)

    is_blamed = Column(Boolean, nullable=False, default=False)

    function_history_id = Column(Integer, ForeignKey('function_history.id'))
    function_history = relationship('FunctionHistory', back_populates='blame_data')

    test_result_id = Column(Integer, ForeignKey('test_result.id'))
    test_result = relationship('TestResult', back_populates='blame_data')

    def __init__(self,
                 function_history: FunctionHistory,
                 test_result: TestResult):
        '''
        Creates a new BlameData instance.
        '''
        self.function_history = function_history
        self.test_result = test_result

    def __repr__(self):
        '''
        Converts the BlameData into a string
        '''
        return ('<BlameData "{function_name}" | {test_result} | {commit} />'
                .format(function_name=self.function.name,
                        test_result=self.test_result.test.name,
                        commit=self.function.commit.commit_id))
