#!/usr/bin/env python3
'''
Object representation a routine, which encompasses functions, methods,
subroutines, etc.  It is a portion of code within a larger program that performs
a specific task and is relatively independent of the remaining code.
'''
import ast
import astor

from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger


class Routine(object):
    '''
    Schema representation of a repository.  Stores the repository and acts as a
    parent relationship across runs and test results.
    '''
    def __init__(self,
                 node: ast.AST,
                 file: str):
        '''
        Creates a new Routine instance.
        '''
        self.node = node
        self.file = file

    def prepend_statement(self, statement):
        '''
        Writes a statement to the beginning of the routine
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
        logger.info('Adding "{statement}" to {file} | {routine_name}@{lineno}'
                    .format(statement=statement,
                            file=self.file,
                            routine_name=self.node.name,
                            lineno=self.node.lineno))
        first_node = self.node.body[0]
        first_line_in_routine = first_node.lineno

        # scoot down one line if the first node is a comment
        first_line_in_routine += 1 if _is_comment(first_node) else 0

        # note that a comment after the function does not seem to have a
        # column offset, and instead returns -1.
        column_offset = (first_node.col_offset if first_node.col_offset != -1
                         else self.node.col_offset + 4)
        indentation = ' ' * column_offset
        statement = indentation + statement + '\n'

        with open(self.file, 'r') as f:
            contents = f.readlines()

        contents.insert(first_line_in_routine - 1, statement)

        with open(self.file, 'w') as f:
            f.writelines(contents)

    def __repr__(self):
        '''
        Converts the Routine into a string
        '''
        return ('<Routine {name} | {file} | {line} />'
                .format(name=self.node.name,
                        file=self.file,
                        line=self.node.lineno))
