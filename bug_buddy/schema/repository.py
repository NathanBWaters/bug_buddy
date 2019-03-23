#!/usr/bin/env python3
'''
The repository model.  Corresponds with a library of code
'''
import ast
import os
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

from bug_buddy.constants import PYTHON_FILE_TYPE, MIRROR_ROOT
from bug_buddy.constants import FILE_TYPES
from bug_buddy.errors import BugBuddyError
from bug_buddy.schema.base import Base
from bug_buddy.schema.function import Function


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
    original_path = Column(String(500), nullable=False)
    _mirror_path = Column('mirror_path', String(500), nullable=False)

    # The set of shell commands to intialize the repository
    initialize_commands = Column(String(500), nullable=False)
    # The set of shell commands to run the tests
    test_commands = Column(String(500), nullable=False)

    # the directory that contains the src files
    src_directory = Column(String(500), nullable=False)

    commits = relationship(
        'Commit',
        back_populates='repository',
        cascade='all, delete, delete-orphan')
    tests = relationship(
        'Test',
        back_populates='repository',
        cascade='all, delete, delete-orphan')
    functions = relationship(
        'Function',
        back_populates='repository',
        cascade='all, delete, delete-orphan')

    repository_files = []

    def __init__(self,
                 name: str,
                 url: str,
                 src_path: str,
                 initialize_commands: str,
                 test_commands: str,
                 src_directory: str,
                 mirror_path: str=None):
        '''
        Creates a new Repository instance.
        '''
        self.name = name
        self.url = url
        self.initialize_commands = initialize_commands
        self.test_commands = test_commands
        self.original_path = os.path.abspath(src_path)
        self._mirror_path = mirror_path or self.mirror_path
        self.src_directory = src_directory

    @property
    def src_path(self):
        '''
        Returns the absolute path to the directory that contains the source
        files
        '''
        src_path = os.path.join(self.path, self.src_directory)
        if not src_path.endswith('/'):
            src_path += '/'
        return src_path

    @property
    def mirror_path(self) -> str:
        '''
        Returns the path to the mirrored repository that is updated in parallel
        to the src_path that the developer is working on
        '''
        return os.path.join(MIRROR_ROOT, self.original_path.split('/')[-1])

    @property
    def path(self):
        '''
        Returns the path of the mirrored repository
        '''
        return self.mirror_path

    def get_src_files(self, filter_file_type=None) -> dict:
        '''
        Returns a list of source files
        '''
        repository_files = []
        for dirname, _, file_names in os.walk(self.src_path):
            for file_name in file_names:
                # client can request a specific file type such as only Python
                # files
                if filter_file_type:
                    if not file_name.endswith(FILE_TYPES[filter_file_type]):
                        continue

                absolute_path = os.path.join(self.src_path,
                                             dirname,
                                             file_name)
                repository_files.append(absolute_path)

        return repository_files

    def __repr__(self):
        '''
        Converts the repository into a string
        '''
        return ('<Repository {id} | {name} | {path} />'
                .format(id=self.id, name=self.name, path=self.path))
