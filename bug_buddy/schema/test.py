#!/usr/bin/env python3
'''
The Test model.  Represents a single test
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Test(Base):
    '''
    Schema representation of a test.
    '''
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='tests')

    test_outputs = relationship('TestOutput', back_populates='test')
    test_results = relationship('TestResult', back_populates='test')

    def __init__(self, repository: Repository, name: str):
        '''
        Creates a new TestResults instance
        '''
        self.name = name
        self.repository = repository
