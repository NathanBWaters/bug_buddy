#!/usr/bin/env python3
'''
Object representation a Blame.  This maps test failures with the lines in the
code base that are to blame.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.line import Line
from bug_buddy.schema.test_result import TestResult


class Blame(Base):
    '''
    Object representation a Blame.  This maps test failures with the lines in
    the code base that are to blame.
    '''
    __tablename__ = 'blame'
    id = Column(Integer, primary_key=True)

    line_id = Column(Integer, ForeignKey('line.id'))
    line = relationship('Line', back_populates='blames')

    test_result_id = Column(Integer, ForeignKey('test_result.id'))
    test_result = relationship('TestResult', back_populates='blames')

    def __init__(self,
                 line: Line,
                 test_result: TestResult):
        '''
        Creates a new Blame instance.
        '''
        self.line = line
        self.test_result = test_result

    def __repr__(self):
        '''
        Converts the Blame into a string
        '''
        return ('<Blame "{line_content}" | {test_result} | {commit} />'
                .format(line_content=self.line.content,
                        test_result=self.test_result.test.name,
                        commit=self.line.commit.commit_id))
