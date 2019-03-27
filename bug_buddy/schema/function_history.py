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
                 last_line: int):
        '''
        Creates a new FunctionHistory instance.
        '''
        self.function = function
        self.commit = commit
        self.source_code = astor.to_source(node)
        self.first_line = first_line
        self.last_line = last_line

    def remove_line(self, line):
        '''
        Removes a particular line from the function
        '''
        with open(self.abs_path, 'r') as f:
            contents = f.readlines()

        content = contents.pop(line - 1)
        logger.info('Removed line: "{}" from {}'
                    .format(content.strip(), self.file_path))

        with open(self.abs_path, 'w') as f:
            f.writelines(contents)

    def prepend_statement(self, statement, offset: int=0):
        '''
        Writes a statement to the beginning of the function
        '''
        def _is_comment(node):
            '''
            Checks to see if the node is a comment.  We need to because we do
            not want to add our statement into the comment.  For some reason,
            comments lineno is the last part of the comment.
            '''
            return (True if hasattr(node, 'value') and
                    isinstance(node.value, ast.Str) else False)

        # Get the first node in the function, which is it's first statement.
        # We will add the statement here
        first_node = self.node.body[0]
        first_line_in_function = first_node.lineno

        # scoot down one function if the first node is a comment
        first_line_in_function += 1 if _is_comment(first_node) else 0
        first_line_in_function += offset

        # note that a comment after the function does not seem to have a
        # column offset, and instead returns -1.
        column_offset = (first_node.col_offset if first_node.col_offset != -1
                         else self.node.col_offset + 4)
        indentation = ' ' * column_offset
        indented_statement = indentation + statement + '\n'

        with open(self.abs_path, 'r') as f:
            contents = f.readlines()

        contents.insert(first_line_in_function - 1, indented_statement)

        with open(self.abs_path, 'w') as f:
            f.writelines(contents)

        logger.info('Added "{statement}" to {file} | {function_name}@{lineno}'
                    .format(statement=statement,
                            file=self.file_path,
                            function_name=self.ast_node.name,
                            lineno=first_line_in_function))

        return first_line_in_function

    @property
    def altered(self):
        '''
        Whether or not the function was modified for this commit
        '''
        corresponding_diff = [diff for diff in self.function.diffs if
                              diff.commit_id == self.commit.commit_id]
        return bool(corresponding_diff)

    def __repr__(self):
        '''
        Converts the FunctionHistory into a string
        '''
        return ('<FunctionHistory {name} | {commit} | '
                '{file_path}@{first_line}-{last_line}/>'
                .format(name=self.function.name,
                        commit=self.commit.commit_id,
                        file_path=self.function.file_path,
                        first_line=self.first_line,
                        last_line=self.last_line))
