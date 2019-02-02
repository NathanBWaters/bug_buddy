#!/usr/bin/env python3
'''
Object representation a Diff.  Pretty simple, only works with additions
currently
'''
from typing import List


class Diff(object):
    '''
    Schema representation of a Diff.  Stores the file_path and line number
    '''
    def __init__(self,
                 content: str,
                 line_number: str,
                 file_path: str,
                 diff_type: str):
        '''
        Creates a new Diff instance.
        '''
        self.content = content
        self.line_number = line_number
        self.file_path = file_path
        self.diff_type = diff_type

    def __repr__(self):
        '''
        Converts the Diff into a string
        '''
        return ('<Diff "{content}" | {file_path} | {line_number} | {diff_type}/>'
                .format(content=self.content,
                        file_path=self.file_path,
                        diff_type=self.diff_type,
                        line_number=self.line_number))
