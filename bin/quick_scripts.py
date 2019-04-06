#!/usr/bin/env python3
'''
I use this file for quick scripting.  For example, updating the database.
'''
import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bug_buddy.db import get_all, session_manager, delete
from bug_buddy.logger import logger
from bug_buddy.schema import Blame


if __name__ == '__main__':
    with session_manager() as session:
        blames = get_all(session, Blame)
        i = 1
        for blame in blames:
            print('On {} of {}'.format(i, len(blames)))
            blame.test = blame.test_result.test
            blame.function = blame.diff.function
            i += 1
