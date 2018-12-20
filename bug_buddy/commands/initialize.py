#!/usr/bin/env python3
'''
Initialize command module.  Required to begin using bug_buddy on a repository.
'''
from bug_buddy.utils.logger import logger
from bug_buddy.database.db import create
from bug_buddy.schema import Repository
from bug_buddy.vcs.git_utils import get_repository_data_from_path


def initialize_repository(path: str):
    '''
    Given a path to a repository, create the repository in the database
    '''
    logger.info('Initializing repository at {}'.format(path))
    url, name = get_repository_data_from_path(path)
    logger.info('Repository name is {} with url {}'.format(name, url))
    test_command = input('Input the command to run the tests for this repo: ')

    # first check to see if the repository already exists
    repository = create(Repository,
                        name=name,
                        url=url,
                        test_command=test_command,
                        path=path)
    logger.info('Your repository has been successfully created!')
