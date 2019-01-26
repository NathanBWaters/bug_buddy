#!/usr/bin/env python3
'''
Object representation of one line of a diff in a git commit.
'''
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class DiffLine(Base):
    '''
    Schema representation of a DiffLine.  It's a single line of a commit diff
    '''
    __tablename__ = 'diff_line'
    id = Column(Integer, primary_key=True)

    # diff_type - whether the line was added or removed.
    diff_type = Column(String(500), nullable=False)

    line_number = Column(Integer, nullable=False)

    # relative path to file
    file_path = Column(String(500), nullable=False)

    # the name of the method that the line is part of
    method = Column(String(500), nullable=False)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='diff_lines')

    blames = relationship('Blame', back_populates='diff_line')

    def __init__(self,
                 commit: Commit,
                 diff_type: str,
                 content: str,
                 method: str,
                 line_number: int,
                 file_path: str):
        '''
        Creates a new DiffLine instance.
        '''
        self.commit = commit
        self.diff_type = diff_type
        self.content = content
        self.method = method
        self.line_number = line_number
        self.file_path = file_path

    def __repr__(self):
        '''
        Converts the DiffLine into a string
        '''
        return ('<DiffLine "{content}" | {file} | {line_number} />'
                .format(content=self.content,
                        file=self.file,
                        line_number=self.line_number))
