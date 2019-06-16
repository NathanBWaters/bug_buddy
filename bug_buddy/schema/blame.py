#!/usr/bin/env python3
'''
Object representation a Blame.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import List

from bug_buddy.schema.base import Base
from bug_buddy.schema.test_result import TestResult
from bug_buddy.schema.diff import Diff


class Blame(Base):
    '''
    Schema representation of a Blame.
    It is a Many to Many relationship between diffs and synthetic diffs.
    '''
    __tablename__ = 'blame'

    id = Column(Integer, primary_key=True)

    diff_id = Column(Integer, ForeignKey('diff.id'))
    diff = relationship('Diff', back_populates='blames')

    test_result_id = Column(Integer, ForeignKey('test_result.id'))
    test_result = relationship('TestResult', back_populates='blames')

    # the diff might not correspond with a function, but if it does then we
    # should store that as well.  We store both the function and tests because
    # it is easier to query for from the database.
    function_id = Column(Integer, ForeignKey('function.id'), nullable=True)
    function = relationship('Function')

    test_id = Column(Integer, ForeignKey('test.id'))
    test = relationship('Test')

    def __init__(self,
                 diff: Diff,
                 test_result: TestResult):
        '''
        Creates a new Blame instance.
        '''
        self.diff = diff
        self.test_result = test_result
        self.function = diff.function
        self.test = test_result.test

    def __repr__(self):
        '''
        Converts the Blame into a string
        '''
        return ('<Blame {test} | {function} />'
                .format(function=self.diff.function,
                        test=self.test_result.test))
