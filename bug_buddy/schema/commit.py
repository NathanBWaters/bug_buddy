#!/usr/bin/env python3
'''
The Commit model.  A record of a particular change in a repository's code
'''
from collections import defaultdict
import numpy
from sqlalchemy import Column, ForeignKey, Integer, String, Binary
from sqlalchemy.orm import relationship, deferred
import sys

from bug_buddy.errors import BugBuddyError
from bug_buddy.constants import TEST_OUTPUT_FAILURE, SYNTHETIC_CHANGE
from bug_buddy.schema.base import Base
from bug_buddy.schema.repository import Repository


class Commit(Base):
    '''
    Schema representation of a change in the code repository.
    '''
    __tablename__ = 'commit'
    id = Column(Integer, primary_key=True)
    commit_id = Column(String, nullable=False)
    branch = Column(String, nullable=False)

    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship('Repository', back_populates='commits')

    # commit_type is an extra metadata around the content of the commit:
    #   1) DEVELOP - a standard change by a developer
    #   2) SYNTHETIC - a synthetic alteration
    #   3) RESET - a commit that resets synthetic alteration commits
    commit_type = Column(String, nullable=False)

    # For synthetic commits, we have a base commit that we break down into
    # smaller commits to determine blame.  The parent commit is the base commit
    # and all smaller commits the revert the diffs of the parent commit have
    # this attribute set.
    # parent_commit_id = Column(Integer, ForeignKey('commit.id'), nullable=True)
    # parent_commit = relationship(
    #     'Commit',
    #     remote_side=[parent_commit_id])
    # child_commits = relationship(
    #     'Commit',
    #     back_populates='parent_commit')

    # this is kind of a hack.  Synthetic commits are created from synthetic
    # diffs.  Retrieving the powerset of a set of diffs from the database is
    # prohibitively expensive.  So we will hash the sorted list of synthetic
    # diff ids and use that hash to retrieve the 'child' commits of a commit.
    synthetic_diff_hash = Column(Integer, nullable=True)

    # this stores the commit tensor for determining which tests are going to
    # fail.  We used the 'deferred' function so that it's loaded lazily and not
    # always brought into memory.  For more information, read:
    #   https://docs.sqlalchemy.org/en/13/orm/loading_columns.html
    _commit_tensor_binary = deferred(
        Column('commit_tensor', Binary, nullable=True))

    test_runs = relationship('TestRun',
                             back_populates='commit',
                             cascade='all, delete, delete-orphan')

    # the corresponding functions histories created in this commit
    function_histories = relationship(
        'FunctionHistory',
        back_populates='commit',
        cascade='all, delete, delete-orphan')

    # the corresponding functions histories created in this commit
    diffs = relationship(
        'Diff',
        back_populates='commit',
        cascade='all, delete, delete-orphan')

    _commit_tensor_numpy = None

    # the raw numpy vector output from the model
    test_result_prediction_data = None

    def __init__(self,
                 repository: Repository,
                 commit_id: str,
                 branch: str,
                 commit_type: str=SYNTHETIC_CHANGE,
                 ):  # Commit
        '''
        Creates a new TestResults instance
        '''
        if not commit_id:
            msg = 'Tried creating commit without a commit_id'
            raise BugBuddyError(msg)

        self.repository = repository
        self.commit_id = commit_id
        self.branch = branch
        self.commit_type = commit_type

    def get_matching_test_result(self, test_result):
        '''
        Retuns the corresponding test output
        '''
        test_run = self.test_runs[0]
        matching_test_results = [
            commit_test_result for commit_test_result in test_run.test_results
            if commit_test_result.test.id == test_result.test.id
        ]
        if not matching_test_results:
            import pdb; pdb.set_trace()
            msg = ('Could not find a matching test result for {} at {}'
                   .format(test_result, self))
            raise BugBuddyError(msg)

        if len(matching_test_results) > 1:
            import pdb; pdb.set_trace()
            msg = ('Found multiple matching test_results for {} at {}'
                   .format(test_result, self))
            raise BugBuddyError(msg)

        return matching_test_results[0]

    def causes_test_failures(self):
        '''
        Returns a bool if the commit causes any test failures
        '''
        return bool(self.test_failures)

    def get_function_histories(self,
                               file_path: str,
                               start_range: int,
                               end_range: int):
        '''
        Retrieves the function histories the match a given file_path and range
        '''
        return [function_history for function_history in self.function_histories
                if (function_history.function.file_path == file_path and
                    function_history.first_line <= start_range and
                    function_history.last_line >= end_range)]

    def get_corresponding_function(self,
                                   file_path: str,
                                   start_range: int,
                                   end_range: int):
        '''
        Retrieves the the function history that most tightly matches a given
        file_path and range

        def func:
            def cat:
                ## edit here
                x = 1
            x = 2

        get_corresponding_function would return 'def cat' for the matching
        function unlike get_function_histories which would return func and cat
        '''
        matching_functions = self.get_function_histories(
            file_path, start_range, end_range)

        matching_function = None
        matching_function_difference = sys.maxsize

        for function_history in matching_functions:
            function_difference = (
                start_range - function_history.first_line +
                function_history.last_line - end_range)

            if function_difference < matching_function_difference:
                matching_function_difference = function_difference
                matching_function = function_history

        return matching_function

    def get_function_for_node(self, node):
        '''
        Given a node it will return the corresponding function history
        '''
        return self.get_corresponding_function(
            file_path=node.file_path,
            start_range=node.first_line,
            end_range=node.last_line
        )

    @property
    def is_synthetic(self):
        '''
        Returns a boolean whether or not the commit is synthetic
        '''
        return self.commit_type == SYNTHETIC_CHANGE

    @property
    def blames(self):
        '''
        Returns the diffs in the commit
        '''
        blames = []
        for test_run in self.test_runs:
            for test_result in test_run.test_results:
                blames.extend(test_result.blames)

        return blames

    @property
    def latest_test_run(self):
        '''
        Returns the most recent test run
        '''
        return self.test_runs[-1]

    @property
    def earliest_test_run(self):
        '''
        Returns the most recent test run
        '''
        return self.test_runs[0]

    @property
    def commit_tensor(self):
        '''
        Returns the commit in tensor form as a numpy array
        '''
        if self._commit_tensor_numpy is None:
            if self._commit_tensor_binary:
                self._commit_tensor_numpy = numpy.fromstring(
                    self._commit_tensor_binary)

        return numpy.reshape(self._commit_tensor_numpy, self.input_shape)

    def needs_blaming(self):
        '''
        Whether the commit needs to undergo the blame process
        '''
        return self.causes_test_failures() and not self.blames

    @property
    def num_tests(self):
        '''
        Returns the number of tests present for the commit
        '''
        return len(self.repository.tests)

    @property
    def functions(self):
        '''
        Returns the functions associated with the commit in order
        '''
        functions = self.repository.functions
        functions.sort(key=lambda func: func.id, reverse=False)
        return functions

    @property
    def test_failures(self):
        '''
        Returns a list of tests that failed in the latest test run for the
        commit
        '''
        test_failures = []
        test_run = self.latest_test_run
        for test_result in test_run.test_results:
            if test_result.status == TEST_OUTPUT_FAILURE:
                test_failures.append(test_result.test)

        return test_failures

    @property
    def num_test_failures(self):
        '''
        Returns a list of tests that failed in the latest test run for the
        commit
        '''
        return len(self.test_failures)

    @property
    def failed_test_results(self):
        '''
        Returns a list of test results of failed tests in the latest test run
        for the commit
        '''
        failed_test_results = []
        test_run = self.latest_test_run
        for test_result in test_run.test_results:
            if test_result.status == TEST_OUTPUT_FAILURE:
                failed_test_results.append(test_result)

        return failed_test_results

    @property
    def num_functions(self):
        '''
        Returns the number of functions present for the commit
        '''
        return len(self.repository.functions)

    @property
    def num_features(self):
        '''
        Returns the number of features for a given function-test union
        '''
        return 3

    @property
    def input_shape(self):
        '''
        Returns the input shape of a commit
        '''
        return (self.num_functions, self.num_tests, self.num_features)

    @property
    def test_result_prediction(self):
        '''
        Returns test result prediction data
        '''
        if self.test_result_prediction_data is None:
            msg = 'Requested prediction data but it does not exist'
            raise BugBuddyError(msg)

        return dict(zip(self.sorted_tests, self.test_result_prediction_data))

    @property
    def sorted_tests(self):
        '''
        Returns the tests sorted by id
        '''
        sorted_tests = self.repository.tests
        sorted_tests.sort(key=lambda test: test.id, reverse=False)
        return sorted_tests

    def has_same_test_result_output(self,
                                    test_result,
                                    status: str=None):
        '''
        Returns true if this commit had the same a test_result output

        @param test_failure
        '''
        matching_test_result = self.get_matching_test_result(test_result)
        if not matching_test_result:
            return False

        if status:
            return (
                matching_test_result.status == status and
                test_result.status == status)

        return matching_test_result.status == test_result.status

    def summary(self, indent=0, blame=True, prediction=True, edits=True):
        '''
        Prints a summary to terminal about this commit
        '''
        print(' ' * indent + str(self))
        print(' ' * indent + 'Number of test runs: {}'.format(len(self.test_runs)))

        if edits:
            print('-' * 10 + ' E D I T S ' + '-' * 10)
            for diff in self.diffs:
                print(' ' * (indent + 2) + str(diff))
            print('\n')

        if blame:
            self.blame_summary()

        if prediction:
            self.prediction_summary()

    def blame_summary(self, indent=0):
        '''
        Prints a summary to terminal about the actual blames of the commit
        '''
        print('-' * 10 + ' A C T U A L ' + '-' * 10)
        function_to_test_map = {}
        for diff in self.diffs:
            function_to_test_map[diff.function] = []

        for test_failure_result in self.earliest_test_run.test_failures:
            for blame in test_failure_result.blames:
                function_to_test_map[blame.function].append(test_failure_result.test)

        ordered_pairing = sorted(function_to_test_map.items(),
                                 key=lambda kv: kv[0].id)
        for function, failed_tests in ordered_pairing:
            print(' ' * (indent + 2) + str(function))
            for failed_test in failed_tests:
                print(' ' * (indent + 4) + str(failed_test))
        print('\n')

    def prediction_summary(self, indent=0):
        '''
        Prints a summary to terminal about the predicted blames of the commit
        '''
        print('-' * 10 + ' P R E D I C T I O N ' + '-' * 10)
        function_to_test_map = defaultdict(list)

        for test_failure_result in self.latest_test_run.test_failures:
            prediction = test_failure_result.predicted_blamed_functions
            for (blamed_function, confidence) in prediction:
                function_to_test_map[blamed_function].append({
                    'test': test_failure_result.test,
                    'confidence': confidence,
                })

        for function, failed_test_data_list in function_to_test_map.items():
            print(' ' * (indent + 2) + str(function))
            for failed_test_data in failed_test_data_list:
                failed_test = failed_test_data['test']
                confidence = failed_test_data['confidence']
                print(' ' * (indent + 4) + str(failed_test) +
                      ' | {}'.format(confidence))
        print('\n')

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Commit {id} | {commit_id} | {repository_name} | {branch} '
                '| {commit_type} />'
                .format(id=self.id,
                        commit_id=self.commit_id,
                        repository_name=self.repository.name,
                        branch=self.branch,
                        commit_type=self.commit_type))
