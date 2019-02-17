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
    Schema representation of a Blame.  Relates one diff to a test failure, but
    of course there can be multiple diffs to be blamed for a test failure.
    '''
    __tablename__ = 'blame'

    id = Column(Integer, primary_key=True)

    diff_id = Column(Integer, ForeignKey('diff.id'))
    diff = relationship('Diff', back_populates='blames')

    test_result_id = Column(Integer, ForeignKey('test_result.id'))
    test_result = relationship('TestResult', back_populates='blames')

    def __init__(self,
                 diff: Diff,
                 test_result: TestResult,
                 is_new: bool):
        '''
        Creates a new Blame instance.
        '''
        self.diff = diff
        self.test_result = test_result

    def __repr__(self):
        '''
        Converts the Blame into a string
        '''
        return ('<Blame {test_result} | {diff}/>'
                .format(diff=self.diff,
                        test_result=self.test_result))
