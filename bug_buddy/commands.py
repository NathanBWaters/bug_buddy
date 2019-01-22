#!/usr/bin/env python3
'''
The argparse subcommands
'''
from bug_buddy.logger import logger
from bug_buddy.db import create, get, session_manager
from bug_buddy.schema import Repository
from bug_buddy.git_utils import (get_repository_url_from_path,
                                 get_repository_name_from_url)
from bug_buddy.synthetic_alterations import generate_synthetic_test_results


def initialize(path: str,
               initialize_commands: str=None,
               test_commands: str=None,
               src_directory: str=None,
               ):
    '''
    Given a path to a repository, create the repository in the database
    '''
    logger.info('Initializing repository at "{}"'.format(path))
    url = get_repository_url_from_path(path)
    name = get_repository_name_from_url(url)
    logger.info('Repository name is "{}" with url "{}"'.format(name, url))
    if not initialize_commands:
        initialize_commands = input('Input the commands to intialize the repo: ')
    if not test_commands:
        test_commands = input('Input the command to run the tests for the repo: ')
    if not src_directory:
        src_directory = input('Input the source directory for your project')

    # first check to see if the repository already exists
    repository = create(
        Repository,
        name=name,
        url=url,
        initialize_commands=initialize_commands,
        test_commands=test_commands,
        src_directory=src_directory,
        path=path)
    logger.info('Your repository "{}" has been successfully created!'
                .format(repository))


def analyze(repository_name: str):
    '''
    Entry-point for the "bugbuddy analyze" command

    @param args
    '''
    print('repository_name: ', repository_name)


def generate(path: str, run_limit: int=None):
    '''
    Entry-point for the "bugbuddy generate" command

    @param repository_name: name of the repository
    '''
    url = get_repository_url_from_path(path)
    with session_manager() as session:
        repository = get(session, Repository, url=url)
        generate_synthetic_test_results(repository, run_limit)
