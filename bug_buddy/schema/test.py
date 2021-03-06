#!/usr/bin/env python3
'''
The Test model.  Represents a single test
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import TEST_OUTPUT_FAILURE
from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Test(Base):
    '''
    Schema representation of a test.  For each TestRun, a Test has a TestResult.
    '''
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    file = Column(String, nullable=False)
    classname = Column(String, nullable=False)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='tests')

    test_results = relationship('TestResult', back_populates='test')

    def __init__(self,
                 repository: Repository,
                 name: str,
                 file: str,
                 classname: str):
        '''
        Creates a new TestResults instance
        '''
        self.name = name
        self.file = file
        self.repository = repository
        self.classname = classname

    @property
    def failing_instances(self):
        '''
        Returns the list of TestResults where the test failed
        '''
        return [test_result for test_result in self.test_results
                if test_result.status == TEST_OUTPUT_FAILURE]

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Test {id} | {name} | {file} | {classname} />'
                .format(id=self.id,
                        name=self.name,
                        file=self.file,
                        classname=self.classname))
