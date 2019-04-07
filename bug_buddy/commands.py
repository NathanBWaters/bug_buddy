#!/usr/bin/env python3
'''
The argparse subcommands
'''
from mock import Mock

# from bug_buddy.ai.predict_test_failures import train
from bug_buddy.brain import synthetic_train
from bug_buddy.cli import is_affirmative
from bug_buddy.logger import logger
from bug_buddy.db import create, get, delete, session_manager
from bug_buddy.schema import Repository, Function, FunctionHistory, Commit
from bug_buddy.git_utils import (
    db_and_git_match,
    delete_bug_buddy_branch,
    get_repository_url_from_path,
    get_repository_name_from_url,
    is_repo_clean,
    set_bug_buddy_branch)
from bug_buddy.source import sync_mirror_repo
from bug_buddy.snapshot import snapshot
from bug_buddy.synthetic_alterations import (
    generate_synthetic_test_results,
    create_synthetic_alterations)
from bug_buddy.watcher import watch


def train_command(src_path: str):
    '''
    Trains a neural network on the available data
    '''
    with session_manager() as session:
        repository = _get_repository_from_src_path(session, src_path)
        logger.info('Training repository: "{}"'.format(repository))
        synthetic_train(repository)


def watch_command(src_path: str, commit_only: bool):
    '''
    Watches a repository and records any changes
    '''
    with session_manager() as session:
        repository = _get_repository_from_src_path(session, src_path)
        watch(repository, commit_only)


def initialize_command(src_path: str,
                       initialize_commands: str=None,
                       test_commands: str=None,
                       src_directory: str=None,
                       commit_only: bool=False,
                       ignored_files: str=''):
    '''
    Initializes a repository
    '''
    with session_manager() as session:
        _initialize_repository(session,
                               src_path,
                               initialize_commands,
                               test_commands,
                               src_directory,
                               commit_only,
                               ignored_files)


def analyze_command(repository_name: str):
    '''
    Entry-point for the "bugbuddy analyze" command

    @param args
    '''
    print('repository_name: ', repository_name)


def generate_command(src_path: str, run_limit: int=None):
    '''
    Entry-point for the "bugbuddy generate" command

    @param src_path: path to the repository
    '''
    with session_manager() as session:
        repository = _get_repository_from_src_path(session, src_path)

        _check_repo_is_clean(repository)

        db_and_git_match(repository)

        logger.info('Creating synthetic results for: {}'.format(repository))

        generate_synthetic_test_results(repository, run_limit)


def delete_command(src_path: str):
    '''
    Entry-point for the "bugbuddy generate" command

    @param src_path: path to the repository
    '''
    url = get_repository_url_from_path(src_path)
    with session_manager() as session:
        repository = get(session, Repository, url=url)

        # make sure you cannot delete the bug_buddy branch
        if repository.name not in ['bug_buddy', 'BugBuddy']:
            msg = ('Would you like to delete the bug_buddy branch for {}?\n'
                   '(y/n)\n'.format(repository))
            should_delete = input(msg)

            if is_affirmative(should_delete):
                logger.info('Deleting bug_buddy branch')
                delete_bug_buddy_branch(repository or Mock(src_path=src_path))

        if repository:
            logger.info('Deleting data from the database')
            delete(session, repository)

        else:
            logger.info('No matching repo found in the database')


def _check_repo_is_clean(repository, path=None):
    '''
    Requires the user to clean the repo if it's has unchecked in files
    '''
    while not is_repo_clean(repository, path=path or repository.path):
        msg = ('You cannot initialize an unclean repository.  Please clean '
               'the repository and then hit enter:\n')
        input(msg)


def _initialize_repository(session,
                           src_path: str,
                           initialize_commands: str=None,
                           test_commands: str=None,
                           src_directory: str=None,
                           commit_only=False,
                           ignored_files=''):
    '''
    Given a src_path to a repository, create the repository in the database
    '''
    logger.info('Initializing repository at "{}"'.format(src_path))
    url = get_repository_url_from_path(src_path)
    name = get_repository_name_from_url(url)
    logger.info('Repository name is "{}" with url "{}"'.format(name, url))
    if not initialize_commands:
        msg = ('Input the commands to intialize the repo (ex. '
               '"source env/bin/activate"): ')
        initialize_commands = input(msg)
    if not test_commands:
        msg = ('Input the command to run the tests for the repo: ')
        test_commands = input(msg)
    if not src_directory:
        msg = ('Input the source directory for your project: ')
        src_directory = input(msg)

    # first check to see if the repository already exists
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
            src_path=src_path,
            ignored_files=ignored_files)

    _check_repo_is_clean(repository, path=repository.original_path)

    # create the mirror repository that BugBuddy primarily works on
    sync_mirror_repo(repository)

    # make sure the mirrored repo is on bug_buddy branch
    set_bug_buddy_branch(repository)

    # Initialize the repository by recording functions and creating synthetic
    # diffs
    if not commit_only:
        snapshot(repository, allow_empty=True)

    session.commit()
    logger.info('Your repository "{}" has been successfully initialized!'
                .format(repository))

    return repository


def _get_repository_from_src_path(session, src_path: str):
    '''
    Returns the repository given a src_path
    '''
    url = get_repository_url_from_path(src_path)
    repository = get(session, Repository, url=url)
    if not repository:
        msg = ('This repository is not in the BudBuddy database, would you '
               'like to initialize the repository?  (y/n)\n'
               .format(src_path))
        should_initialize = input(msg)
        if is_affirmative(should_initialize):
            repository = _initialize_repository(session, src_path)
        else:
            logger.info('No worries!')

    return repository
