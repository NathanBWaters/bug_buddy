#!/usr/bin/env python3
'''
Command to create synthetic test results
'''
import logging

from bug_buddy.alter import generate_synthetic_test_results
from bug_buddy.database.db import get
from bug_buddy.schema import Repository, TestRun


def generate(repository_name: str, path: str, run_limit: int=None):
    '''
    Entry-point for the "bugbuddy generate" command

    @param repository_name: name of the repository
    '''
    repository = get(Repository, name=repository_name)
    generate_synthetic_test_results(repository, run_limit)
