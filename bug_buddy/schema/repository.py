#!/usr/bin/env python3
'''
The repository model.  Corresponds with a library of code
'''
import os
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import FILE_TYPES
from bug_buddy.errors import BugBuddyError
from bug_buddy.schema.base import Base
from bug_buddy.vcs.git_utils import get_name_from_url
# from bug_buddy.schema.commit import Commit


class Repository(Base):
    '''
    Schema representation of a repository.  Stores the repository and acts as a
    parent relationship across runs and test results.
    '''
    __tablename__ = 'repository'
    id = Column(Integer, primary_key=True)

    # Shorthand name of the repository
    name = Column(String(500), nullable=False)

    # the url for the source code, i.e. Gitlab, Github, Bitbucket, etc.
    url = Column(String(500), nullable=False)

    # path to the project.
    # TODO: It does not make sense to store the path in the database,
    # considering it will very likely be different from machine to machine.
    # This is definitely tech debt, and the path needs to be more dynamic in the
    # future.  Or it should be saved on a per-machine basis in the database.
    path = Column(String(500), nullable=False)

    # The set of shell commands to run the tests
    test_command = Column(String(500), nullable=False)

    commits = relationship(
        'Commit',
        back_populates='repository',
        cascade="all, delete, delete-orphan")
    tests = relationship(
        'Test',
        back_populates='repository',
        cascade="all, delete, delete-orphan")

    repository_files = []

    def __init__(self,
                 name: str,
                 url: str,
                 path: str,
                 test_command: str):
        '''
        Creates a new Repository instance.
        '''
        self.name = name
        self.url = url
        self.test_command = test_command
        self.path = path

    @property
    def path(self):
        '''
        Path to the repository.  If it does not already exist, it will clone
        it locally.
        '''
        if not self._path:
            self._path = self.clone_locally()

        return self._path

    def clone_locally(self):
        '''
        Clones the repository locally

        @return: path to the cloned repository
        '''
        print('Implement clone_locally')
        assert False

    def get_files(self, commit=None, filter_file_type=None) -> dict:
        '''
        Returns a list of files
        '''
        if commit:
            self.set_repo_to_commit(commit)

        if not self.repository_files:
            self.save_files()

        repository_files = []
        for dirname, _, file_names in os.walk(self.repository_path):
            for file_name in file_names:
                # client can request a specific file type such as only Python
                # files
                if filter_file_type:
                    if not file_name.endswith(FILE_TYPES[filter_file_type]):
                        continue

                absolute_path = os.path.join(self.repository_path,
                                             dirname,
                                             file_name)
                repository_files.append(absolute_path)

        return repository_files

    def set_repo_to_commit(self, commit):
        '''
        Sets the respoitory to match a commit
        '''
        print('Implement set_repo_to_commit')
        assert False

    def reset_repo(self):
        '''
        Resets the repository to master/latest
        '''
        print('Implement reset_repo')
        assert False

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Repository {name} | {path} />'
                .format(name=self.name, path=self.path))
