#!/usr/bin/env python3
'''
The Commit model.  A record of a particular change in a repository's code
'''

from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Commit(Base):
    '''
    Schema representation of a change in the code repository.
    '''
    __tablename__ = 'commit'
    id = Column(Integer, primary_key=True)
    commit_id = Column(String, nullable=False)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='commits')

    test_runs = relationship('TestRun', back_populates='commit')

    def __init__(self, repository: Repository, commit_id):
        '''
        Creates a new TestResults instance
        '''
        self.commit_id = commit_id
        self.repository = repository
