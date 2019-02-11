#!/usr/bin/env python3
'''
Object representation a Diff.  Pretty simple, only works with additions
currently
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import List

from bug_buddy.constants import DIFF_ADDITION
from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class Diff(Base):
    '''
    Schema representation of a Diff.  Stores the file_path and line number
    '''
    __tablename__ = 'diff'

    id = Column(Integer, primary_key=True)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='diffs')

    content = Column(String, nullable=False)
    first_line = Column(Integer, nullable=False)
    last_line = Column(Integer, nullable=False)
    file_path = Column(String, nullable=False)
    diff_type = Column(String, nullable=False)

    # The patch stores the diff information in a universally accessible way.
    # When we are reverting a diff, we can use the patch to revert in a
    # less error-prone way
    patch = Column(String, nullable=False)

    def __init__(self,
                 commit: Commit,
                 content: str,
                 first_line: str,
                 last_line: str,
                 patch: str,
                 file_path: str,
                 diff_type: str):
        '''
        Creates a new Diff instance.
        '''
        self.commit = commit
        self.content = content
        self.first_line = first_line
        self.last_line = last_line
        self.patch = patch
        self.file_path = file_path
        self.diff_type = diff_type

    @property
    def size_difference(self):
        '''
        # TODO - huge assumption that the diff is only 1 size right now and that
        the it is an ADDITION of a change
        '''
        is_positive = 1 if self.diff_type == DIFF_ADDITION else -1
        return is_positive * 1

    def revert(self):
        '''
        Given a diff, it will revert the code it altered in the source code
        '''
        assert False, 'you need to implement revert for Diffs'

    def __repr__(self):
        '''
        Converts the Diff into a string
        '''
        return ('<Diff "{content}" | {file_path} | {first_line}-{last_line}'
                ' | {diff_type}/>'
                .format(content=self.content,
                        file_path=self.file_path,
                        diff_type=self.diff_type,
                        first_line=self.first_line,
                        last_line=self.last_line))
