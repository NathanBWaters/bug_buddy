#!/usr/bin/env python3
'''
Represents a function's history for each commit.

NOT CURRENTLY USED
'''
import ast
import astor
import pickle
from sqlalchemy import Column, ForeignKey, Integer, Boolean, String
from sqlalchemy.orm import relationship

from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema.base import Base
from bug_buddy.schema.function import Function
from bug_buddy.schema.commit import Commit


class FunctionHistory(Base):
    '''
    Schema representation of a function's history.
    '''
    __tablename__ = 'function_history'
    id = Column(Integer, primary_key=True)

    # the first and last line number of the function at the commit
    first_line = Column(Integer, nullable=False)
    last_line = Column(Integer, nullable=False)

    # whether or not the function was altered in the commit
    altered = Column(Boolean, nullable=False)

    # a serialized or 'pickled' form of the AST node
    source_code = Column(String, nullable=False)

    function_id = Column(Integer, ForeignKey('function.id'))
    function = relationship('Function', back_populates='function_history')

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='function_histories')

    test_maps = relationship(
        'FunctionToTestLink',
        back_populates='function_history',
        cascade='all, delete, delete-orphan')

    @property
    def name(self):
        '''
        Return the name of the corresponding function
        '''
        return self.function.name

    def __init__(self,
                 function: Function,
                 commit: Commit,
                 node,
                 first_line: int,
                 last_line: int,
                 altered: bool):
        '''
        Creates a new FunctionHistory instance.
        '''
        self.function = function
        self.commit = commit
        self.source_code = astor.to_source(node)
        self.first_line = first_line
        self.last_line = last_line
        self.altered = altered

    def __repr__(self):
        '''
        Converts the FunctionHistory into a string
        '''
        return ('<FunctionHistory {name} | {commit} />'
                .format(name=self.function.name,
                        commit=self.commit.commit_id))
