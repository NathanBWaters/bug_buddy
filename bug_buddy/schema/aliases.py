'''
Selection of typing aliases based on our schema models
'''
from typing import List

from bug_buddy.schema.diff import Diff
from bug_buddy.schema.function import Function
from bug_buddy.schema.function_history import FunctionHistory

# aliases for typing
DiffList = List[Diff]
FunctionList = List[Function]
FunctionHistoryList = List[FunctionHistory]
