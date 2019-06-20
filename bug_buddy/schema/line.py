#!/usr/bin/env python3
'''
Object representation of one line of code for a project at a particular commit
'''
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class Line(object):
    '''
    Schema representation of a Line.  It's a single line of a project.  If the
    content is changed, it becomes a new line.  Therefore it's representation
    stays static as long as it's content stays the same and it's part of the
    same method.

    At the beginning of a project, a line in the database is made for each line
    in the project.  Then we just add the diffs.
    '''
    __tablename__ = 'line'
    id = Column(Integer, primary_key=True)

    # the content of the line
    content = Column(String(500), nullable=False)

    line_number = Column(Integer, nullable=False)

    # relative path to file
    file_path = Column(String(500), nullable=False)

    # the name of the method that the line is part of
    method = Column(String(500), nullable=False)

    # the first commit that this line was a part of
    starting_commit_id = Column(Integer,
                                ForeignKey('commit.id'),
                                nullable=False)
    starting_commit = relationship('Commit',
                                   back_populates='added_lines',
                                   foreign_keys=[starting_commit_id])

    # the last commit this line was a part of
    ending_commit_id = Column(Integer, ForeignKey('commit.id'), nullable=True)
    ending_commit = relationship('Commit',
                                 back_populates='removed_lines',
                                 foreign_keys=[ending_commit_id])

    blames = relationship('Blame', back_populates='line')

    def __init__(self,
                 starting_commit: Commit,
                 content: str,
                 method: str,
                 line_number: int,
                 file_path: str):
        '''
        Creates a new Line instance.
        '''
        self.starting_commit = starting_commit
        self.content = content
        self.method = method
        self.line_number = line_number
        self.file_path = file_path

    def __repr__(self):
        '''
        Converts the Line into a string
        '''
        return ('<Line "{content}" | {file} | {line_number} />'
                .format(content=self.content,
                        file=self.file,
                        line_number=self.line_number))
