'''
Tests bug_buddy.collection
'''
import os
from mock import Mock

from . import BugBuddyTest
from bug_buddy import collection


class TestCollection(BugBuddyTest):
    '''
    Tests bug_buddy.collection
    '''

    def test_pytest_create_results_from_junit_xml(self):
        '''
        Tests a pytest junit output
        '''
        collection.create_results_from_junit_xml(
            self.example_repo,
            self.example_test_run)

        self.assertEquals(len(self.example_test_run.test_results), 6)
