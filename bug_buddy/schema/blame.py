#!/usr/bin/env python3
'''
Object representation a Blame.  This maps test failures with the lines in the
code base that are to blame.
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base


class Blame(Base):
    '''
    Object representation a Blame.  This maps test failures with the lines in the
    code base that are to blame.
    '''
    __tablename__ = 'blame'
    id = Column(Integer, primary_key=True)

    diff_line_id = Column(Integer, ForeignKey('diff_line.id'))
    diff_line = relationship('DiffLine', back_populates='blames')

    test_result_id = Column(Integer, ForeignKey('test_result.id'))
    test_result = relationship('TestResult', back_populates='blames')

    def __init__(self,
                 diff_line: str,
                 test_result: str):
        '''
        Creates a new Blame instance.
        '''
        self.diff_line = diff_line
        self.test_result = test_result

    def __repr__(self):
        '''
        Converts the Blame into a string
        '''
        return ('<Blame "{line_content}" | {test_result} | {commit} />'
                .format(line_content=self.diff_line.content,
                        test_result=self.test_result.test.name,
                        commit=self.diff_line.commit.commit_id))
