#!/usr/bin/env python3
'''
The Commit model.  A record of a particular change in a repository's code
'''
import numpy
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean, Binary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
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

    _commit_tensor_binary = Column('commit_tensor', Binary, nullable=True)

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
    def test_failures(self):
        '''
        Returns the number of tests present for the commit
        '''
        test_failures = []
        for test_run in self.test_runs:
            for test_result in test_run.test_results:
                if test_result.status == TEST_OUTPUT_FAILURE:
                    test_failures.append(test_result.test)

        return test_failures

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
