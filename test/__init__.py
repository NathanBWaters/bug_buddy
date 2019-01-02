'''
Stores the base class for running unit tests that all bug_buddy tests inherit
from
'''
import os
import sys
import unittest

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(dir_path, '..'))

from bug_buddy.db import get_or_create
from bug_buddy.schema import Repository, Commit, TestRun


class BugBuddyTest(unittest.TestCase):
    '''
    Base class for bug_buddy unit tests
    '''
    def setUp(self):
        '''
        Set up variables used across tests
        '''
        self.TestDir = os.path.dirname(os.path.realpath(__file__))
        self.DataDir = os.path.join(self.TestDir, 'testenv')

        self.populate_example_data()

    def populate_example_data(self):
        '''
        Populates the sqlite database with example data
        '''
        self.example_repo, _ = get_or_create(
            Repository,
            name='example_repo',
            src_directory='src_dir',
            url='https://github.com/exe/example_repo',
            path='/path/to/fake/dir',
            initialize_commands='source env/bin/activate',
            test_commands='pytest')
        self.example_commit, _ = get_or_create(
            Commit,
            repository=self.example_repo,
            commit_id='e657b562175400d499640f5cd8b3449700b5f36d')
        self.example_test_run, _ = get_or_create(
            TestRun,
            commit=self.example_commit,
            output_file=os.path.join(self.DataDir,
                                     'example_results/pytest_junit.xml'))
