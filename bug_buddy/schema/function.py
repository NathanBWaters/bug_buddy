#!/usr/bin/env python3
'''
Object representation of a function.  It is a portion of code within a larger
program that performs a specific task.  Who am I kididng we all know what a
function is.
'''
import ast
import astor
import os
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema.base import Base


class Function(Base):
    '''
    Schema representation of a function.
    '''
    __tablename__ = 'function'
    id = Column(Integer, primary_key=True)

    # the content of the function
    name = Column(String(500), nullable=False)

    # relative path to file from the root of the repository
    file_path = Column(String(500), nullable=False)

    # relative path to file
    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='functions')

    function_history = relationship(
        'FunctionHistory',
        back_populates='function',
        cascade='all, delete, delete-orphan')

    diffs = relationship(
        'Diff',
        back_populates='function',
        cascade='all, delete, delete-orphan')

    def __init__(self,
                 repository,  # Repository - need to figure out typing for
                              # in cases where they both refer to each other
                 name: str,
                 file_path: str):
        '''
        Creates a new Function instance.
        '''
        self.repository = repository
        self.name = name
        self.file_path = file_path

    @property
    def latest_history(self):
        '''
        Returns the most recent history
        '''
        return self.function_history[0]

    @property
    def abs_path(self):
        '''
        Returns the absolute path
        '''
        return os.path.join(self.repository.path, self.file_path)

    def __repr__(self):
        '''
        Converts the Function into a string
        '''
        return ('<Function {name} | {file} />'
                .format(name=self.name,
                        file=self.file_path))
