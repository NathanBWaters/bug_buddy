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

    # the original line number of the function when it was first introduced
    line_number = Column(Integer, nullable=False)

    # relative path to file from the root of the repository
    path = Column(String(500), nullable=False)

    # relative path to file
    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='functions')

    function_history = relationship(
        'FunctionHistory',
        back_populates='function',
        cascade='all, delete, delete-orphan')

    def __init__(self,
                 repository,  # Repository - need to figure out typing for
                              # in cases where they both refer to each other
                 node: ast.AST,
                 path: str):
        '''
        Creates a new Function instance.
        '''
        self.repository = repository
        self.node = node
        self.name = node.name
        self.line_number = node.lineno
        self.path = path

    @property
    def ast_node(self):
        '''
        Returns the AST node representation of the function.  If it is not
        already loaded, it will retrieve it from the source file
        '''
        if not self.node:
            msg = 'you need to load the ast node for {}'.format(self.name)
            raise Exception(msg)

        return self.node

    @property
    def first_line(self):
        '''
        Returns the first line in the function
        '''
        return self.ast_node.body[0].lineno

    @property
    def last_line(self):
        '''
        Returns the last line in the function
        '''
        return self.ast_node.body[-1].lineno

    @property
    def abs_path(self):
        '''
        Returns the absolute path
        '''
        return os.path.join(self.repository.path, self.path)

    def prepend_statement(self, statement):
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
        logger.info('Adding "{statement}" to {file} | {function_name}@{lineno}'
                    .format(statement=statement,
                            file=self.file_path,
                            function_name=self.ast_node.name,
                            lineno=self.ast_node.lineno))
        first_node = self.ast_node.body[0]
        first_function_in_function = first_node.lineno

        # scoot down one function if the first node is a comment
        first_function_in_function += 1 if _is_comment(first_node) else 0

        # note that a comment after the function does not seem to have a
        # column offset, and instead returns -1.
        column_offset = (first_node.col_offset if first_node.col_offset != -1
                         else self.ast_node.col_offset + 4)
        indentation = ' ' * column_offset
        statement = indentation + statement + '\n'

        with open(self.file_path, 'r') as f:
            contents = f.readlines()

        contents.insert(first_function_in_function - 1, statement)

        with open(self.file_path, 'w') as f:
            f.writelines(contents)

    def __repr__(self):
        '''
        Converts the Function into a string
        '''
        return ('<Function {name} | {file} | {function} />'
                .format(name=self.ast_node.name,
                        file=self.file_path,
                        function=self.ast_node.lineno))
