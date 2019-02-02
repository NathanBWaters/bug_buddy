#!/usr/bin/env python3
'''
Represents a function's history for each commit.
'''
import ast
import astor
from sqlalchemy import Column, ForeignKey, Integer, Boolean
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

    # whether or not the function was altered in this commit
    altered = Column(Boolean, nullable=False)

    function_id = Column(Integer, ForeignKey('function.id'))
    function = relationship('Function', back_populates='function_history')

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='function_histories')

    blame_data = relationship(
        'BlameData',
        back_populates='function_history',
        cascade='all, delete, delete-orphan')

    def __init__(self,
                 function: Function,
                 commit: Commit,
                 altered: bool):
        '''
        Creates a new FunctionHistory instance.
        '''
        self.function = function
        self.commit = commit
        self.altered = altered

    def __repr__(self):
        '''
        Converts the FunctionHistory into a string
        '''
        return ('<FunctionHistory {name} | {file} | {function} />'
                .format(name=self.ast_node.name,
                        file=self.file_path,
                        function=self.ast_node.lineno))
