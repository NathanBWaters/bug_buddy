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

    def __init__(self,
                 repository,  # Repository - need to figure out typing for
                              # in cases where they both refer to each other
                 node: ast.AST,
                 file_path: str):
        '''
        Creates a new Function instance.
        '''
        self.repository = repository
        self.node = node
        self.name = node.name
        self.file_path = file_path

    @property
    def ast_node(self):
        '''
        Returns the AST node representation of the function.  If it is not
        already loaded, it will retrieve it from the source file
        '''
        if not self.loaded_node():
            msg = 'you need to load the ast node for {}'.format(self.name)
            raise Exception(msg)

        return self.node

    def loaded_node(self):
        '''
        Returns whether or not the AST node for the function has been loaded
        '''
        return hasattr(self, 'node')

    @property
    def latest_history(self):
        '''
        Returns the most recent history
        '''
        return self.function_history[0]

    def has_line_info(self):
        '''
        Returns whether or not the function can return representative line
        information
        '''
        return self.loaded_node() or len(self.function_history) > 0

    @property
    def first_line(self):
        '''
        Returns the first line in the function
        '''
        if self.has_line_info():
            if self.loaded_node():
                return self.ast_node.lineno

            return self.latest_history.first_line

        return -1

    def update_given_diff(self):
        '''
        Given a diff, it will update it's internal structures accordingly
        '''

    @property
    def last_line(self):
        '''
        Returns the last line in the function
        '''
        if self.has_line_info():
            if self.loaded_node():
                return self.ast_node.body[-1].lineno

            return self.latest_history.last_line

        return -1

    @property
    def abs_path(self):
        '''
        Returns the absolute path
        '''
        return os.path.join(self.repository.path, self.file_path)

    def remove_line(self, line):
        '''
        Removes a particular line from the function
        '''
        with open(self.abs_path, 'r') as f:
            contents = f.readlines()

        content = contents.pop(line - 1)
        logger.info('Removed line: "{}"'.format(content))

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
        first_node = self.ast_node.body[0]
        first_line_in_function = first_node.lineno

        # scoot down one function if the first node is a comment
        first_line_in_function += 1 if _is_comment(first_node) else 0
        first_line_in_function += offset

        # note that a comment after the function does not seem to have a
        # column offset, and instead returns -1.
        column_offset = (first_node.col_offset if first_node.col_offset != -1
                         else self.ast_node.col_offset + 4)
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

    def __repr__(self):
        '''
        Converts the Function into a string
        '''
        return ('<Function {name} | {file} | {first_line}-{last_line} />'
                .format(name=self.name,
                        file=self.file_path,
                        first_line=self.first_line,
                        last_line=self.last_line))
