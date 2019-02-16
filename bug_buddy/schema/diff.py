#!/usr/bin/env python3
'''
Object representation a Diff.  Pretty simple, only works with additions
currently
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import List

from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Diff(Base):
    '''
    Schema representation of a Diff.  Stores the file_path and line number
    '''
    __tablename__ = 'diff'

    id = Column(Integer, primary_key=True)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='diffs')

    first_line = Column(Integer, nullable=False)
    last_line = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)

    # The patch stores the diff information in a universally accessible way.
    # When we are reverting a diff, we can use the patch to revert in a
    # less error-prone way
    patch = Column(String, nullable=False)

    def __init__(self,
                 repository: Repository,
                 first_line: str,
                 last_line: str,
                 patch: str,
                 file_path: str):
        '''
        Creates a new Diff instance.
        '''
        self.repository = repository
        self.first_line = first_line
        self.last_line = last_line
        self.patch = patch
        self.file_path = file_path

    @property
    def size_difference(self):
        '''
        # TODO - huge assumption that the diff is only 1 size right now and that
        the it is an ADDITION of a change
        '''
        return 1

    @property
    def repository(self):
        '''
        # TODO - huge assumption that the diff is only 1 size right now and that
        the it is an ADDITION of a change
        '''
        return self.repository.repository

    def __repr__(self):
        '''
        Converts the Diff into a string
        '''
        return ('<Diff {file_path} | {first_line}-{last_line} />'
                .format(file_path=self.file_path,
                        first_line=self.first_line,
                        last_line=self.last_line))
