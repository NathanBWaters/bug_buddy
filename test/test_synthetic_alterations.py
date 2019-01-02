'''
Tests bug_buddy.synthetic_alterations.py
'''
import os
from mock import Mock

from . import BugBuddyTest
from bug_buddy import synthetic_alterations


class TestSyntheticAlterations(BugBuddyTest):
    '''
    Tests bug_buddy.synthetic_alterations.py
    '''

    def test_edit_random_routines(self):
        '''
        Tests altering random functions.  We make sure that 'assert True' is
        added to each routine in the testenv/example_repo and make sure
        it matches the files in the altered_example_repo baseline
        '''
        # test 1
        baseline_altered_example_dir = os.path.join(self.DataDir,
                                                    'altered_example_repo')

        num_routines = len(synthetic_alterations.get_routines_from_repo(
            self.example_repo))
        # edit the repository
        synthetic_alterations.edit_random_routines(
            self.example_repo, num_edits=num_routines)
