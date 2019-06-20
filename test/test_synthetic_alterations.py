'''
Tests bug_buddy.synthetic_alterations.py
'''
import os
from mock import Mock, patch

from . import BugBuddyTest
from bug_buddy import synthetic_alterations
from bug_buddy import git_utils


class TestSyntheticAlterations(BugBuddyTest):
    '''
    Tests bug_buddy.synthetic_alterations.py
    '''
    def test_get_routines_from_file(self):
        '''
        Tests synthetic_alterations.get_routines_from_file
        '''
        fun_file = os.path.join(self.example_repo.src_path, 'fun.py')
        routines = synthetic_alterations.get_routines_from_file(
            self.example_repo, fun_file)
        expected = ['present_wrapper',
                    'pets_are_great',
                    'no_comment',
                    'wrapper',
                    '_can_we_handle_func_within_func']
        self.assertEqual(expected, [routine.node.name for routine in routines])

        moves_file = os.path.join(self.example_repo.src_path, 'moves.py')
        routines = synthetic_alterations.get_routines_from_file(
            self.example_repo, moves_file)
        expected = ['__init__', 'much', 'progress']
        self.assertEqual(expected, [routine.node.name for routine in routines])

        init_file = os.path.join(self.example_repo.src_path, '__init__.py')
        routines = synthetic_alterations.get_routines_from_file(
            self.example_repo, init_file)
        expected = []
        self.assertEqual(expected, [routine.node.name for routine in routines])

    def test_get_routines_from_repo(self):
        '''
        Tests getting all the routines from a repository
        '''
        routines = synthetic_alterations.get_routines_from_repo(
            self.example_repo)
        expected = ['present_wrapper',
                    'pets_are_great',
                    'no_comment',
                    'wrapper',
                    '_can_we_handle_func_within_func',
                    '__init__',
                    'much',
                    'progress']
        expected.sort()

        routine_names = [routine.node.name for routine in routines]
        routine_names.sort()
        self.assertEqual(expected, routine_names)

    @patch('bug_buddy.synthetic_alterations.random')
    def test_edit_random_routines(self, mock_random):
        '''
        Tests altering random functions.  We make sure that 'assert True' is
        added to each routine in the testenv/example_repo and make sure
        it matches the files in the altered_example_repo baseline
        '''
        # in case we didn't clean up the example repo in another test
        git_utils.revert_unstaged_changes(self.example_repo)

        # make sure the we only add benign statements
        mock_random.randint.return_value = 0

        # get the number of routines, we want to edit all of them
        num_routines = len(synthetic_alterations.get_routines_from_repo(
            self.example_repo))

        # edit the repository
        synthetic_alterations.edit_random_routines(
            self.example_repo, num_edits=num_routines)

        # make sure there were edits
        self.assertFalse(git_utils.is_repo_clean(self.example_repo))

        # compare what was altered vs what we expected to be altered
        baseline_altered_example_dir = os.path.join(
            self.DataDir,
            'altered_example_repo/src_dir')
        for file_name in os.listdir(self.example_repo.src_path):
            baseline_path = os.path.join(baseline_altered_example_dir,
                                         file_name)
            altered_path = os.path.join(self.example_repo.src_path, file_name)

            with open(baseline_path) as f:
                baseline_content = f.read()

            with open(altered_path) as f:
                altered_content = f.read()

            self.assertEqual(baseline_content, altered_content)

        git_utils.revert_unstaged_changes(self.example_repo)
