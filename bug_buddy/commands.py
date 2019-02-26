#!/usr/bin/env python3
'''
The argparse subcommands
'''
from mock import Mock

from bug_buddy.logger import logger
from bug_buddy.db import create, get, delete, session_manager
from bug_buddy.schema import Repository
from bug_buddy.git_utils import (delete_bug_buddy_branch,
                                 get_repository_url_from_path,
                                 get_repository_name_from_url)
from bug_buddy.snapshot import snapshot_initialization
from bug_buddy.synthetic_alterations import generate_synthetic_test_results


def _confirmed(user_response):
    '''
    Utility for determining if the user responded positively or not
    '''
    return user_response == 'y' or user_response == 'yes'


def train(path: str):
    '''
    Trains a neural network on the available data
    '''
    pass


def initialize(path: str,
               initialize_commands: str=None,
               test_commands: str=None,
               src_directory: str=None):
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
    with session_manager() as session:
        repository = get(session, Repository, url=url)
        if not repository:
            repository = create(
                session,
                Repository,
                name=name,
                url=url,
                initialize_commands=initialize_commands,
                test_commands=test_commands,
                src_directory=src_directory,
                path=path)

        snapshot_initialization(repository)

        session.commit()
        logger.info('Your repository "{}" has been successfully initialized!'
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

    @param path: path to the repository
    '''
    url = get_repository_url_from_path(path)
    with session_manager() as session:
        repository = get(session, Repository, url=url)
        if not repository:
            msg = ('Cannot find a corresponding repository for "{}".\nWould you '
                   'like to initialize the repository?  (y/n)\n'.format(path))
            should_initialize = input(msg)

            if _confirmed(should_initialize):
                initialize(path)

            else:
                exit('Ok, not initializing that repository')

        logger.info('Creating synthetic results for: {}'.format(repository))

        generate_synthetic_test_results(repository, run_limit)


def delete_repository(path: str):
    '''
    Entry-point for the "bugbuddy generate" command

    @param path: path to the repository
    '''
    url = get_repository_url_from_path(path)
    with session_manager() as session:
        repository = get(session, Repository, url=url)

        msg = ('Are you sure you want to delete {}?\n'
               'This will delete the bug_buddy branch and all data associated '
               'with this project from the database.  (y/n)\n'
               .format(path))
        should_initialize = input(msg)

        if _confirmed(should_initialize):
            logger.info('Deleting bug_buddy branch')
            delete_bug_buddy_branch(repository or Mock(path=path))

        if repository:
            logger.info('Deleting data from the database')
            delete(session, repository)

        else:
            logger.info('No matching repo found in the database')
