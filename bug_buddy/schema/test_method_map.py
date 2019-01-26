#!/usr/bin/env python3
'''
The line model.  Corresponds with a single line of code in the diff.  Stores
various information about the line.  This becomes the vector that is fed into
the machine learning model.
'''
import numpy
import os
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import FILE_TYPES
from bug_buddy.errors import BugBuddyError
from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit


class TestRoutineMap(Base):
    '''
    Schema representation mapping the relationship between each Test and each
    Routine.
    '''
    __tablename__ = 'test_routine_map'
    id = Column(Integer, primary_key=True)

    # relative path to file
    file_path = Column(String(500), nullable=False)

    # the name of the method that the line is part of
    method_name = Column(String(500), nullable=False)

    # The associated commit for this line
    test_id = Column(Integer, ForeignKey('test.id'))
    test = relationship('Test', back_populates='routine_map')

    # the number of times the routine was altered and the test began failing.
    # Note that correlation does not imply causation, but this is a heuristic.
    num_correlating_failures = Column(Integer, nullable=False, default=0)
    # the number of times the routine was altered and the test began passing.
    # Note that correlation does not imply causation, but this is a heuristic.
    num_correlating_passes = Column(Integer, nullable=False, default=0)

    # the distance between the test and the routine over the number 1.
    # Examples:
    #   1) if the test directly calls this function, the distance is 1
    #   2) if the test calls a function which calls this function, the
    #      distance is 1/2
    #   3) if the test does not call this function, the distance is -1
    distance = Column(Integer, nullable=False, default=0)

    def __init__(self,
                 name: str,
                 url: str,
                 path: str,
                 initialize_commands: str,
                 test_commands: str,
                 src_directory: str):
        '''
        Creates a new Line instance.
        '''
        self.name = name
        self.url = url
        self.initialize_commands = initialize_commands
        self.test_commands = test_commands
        self.path = path
        self.src_directory = src_directory

    @property
    def vector(self):
        '''
        Converts the line into a vector
        '''
        return numpy.asarray([

        ])

    def __repr__(self):
        '''
        Converts the Line object into a string
        '''
        return ('<Line {id} | "{content}" | {commit_id} | {file} @ {line} />'
                .format(id=self.id,
                        content=self.content,
                        commit_id=self.commit.commit_id,
                        file=self.file,
                        line=self.line_number))
