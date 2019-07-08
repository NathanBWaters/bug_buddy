'''
The watcher records a developer's changes
'''
import os
import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from bug_buddy.brain import Brain, predict_blame
from bug_buddy.db import get, session_manager, Session
from bug_buddy.git_utils import (
    go_to_commit,
    is_repo_clean,
    set_bug_buddy_branch)
from bug_buddy.logger import logger
from bug_buddy.runner import run_all_tests
from bug_buddy.schema import Repository, Commit
from bug_buddy.snapshot import snapshot
from bug_buddy.scorecard import Scorecard
from bug_buddy.source import sync_mirror_repo


class ChangeWatchdog(PatternMatchingEventHandler):
    '''
    Is notified every time an event occurs on the fileystem and will snapshot
    the change
    '''
    def __init__(self, repository, commit_only: bool):
        '''
        Create a ChangeWatchdog instance
        '''
        super().__init__()

        self.repository = repository
        self.commit_only = commit_only

        self.score_card = Scorecard()
        self.score_card.render(clear=True)

        self.brain = Brain(repository)

    def on_any_event(self, event):
        '''
        Catches all events
        '''
        if '/.' in event.src_path:
            return

        updated_file = os.path.relpath(event.src_path,
                                       self.repository.original_path)
        if (not updated_file or updated_file in self.repository.ignored_files or
                not updated_file.endswith('.py')):
            return

        # we have to recreate the repository in this thread for Sqlite
        with session_manager() as session:
            repository = get(session, Repository, id=self.repository.id)
            # logger.info('Syncing updates')
            # Copy the change over to the mirror repository
            sync_mirror_repo(repository)

            if not is_repo_clean(self.repository):
                logger.info('Valid change event: {}'.format(event))

                # make sure the repository is on the bug_buddy branch
                start = time.time()
                commit = snapshot(repository, commit_only=self.commit_only)
                total_time = time.time() - start
                logger.info('Completed snapshot of {commit} in {m}m {s}s'
                            .format(commit=commit,
                                    m=total_time / 60,
                                    s=total_time % 60))
                session.commit()

                # display the results in the cli output
                # self.score_card.render(commit)

            else:
                logger.info('Nothing was changed')


def watch(repository: Repository, commit_only: bool):
    '''
    Watches the repository's filesystem for changes and records the changes.
    It also notifies the user when there is an update in the test output.
    '''
    session = Session.object_session(repository)
    set_bug_buddy_branch(repository)
    logger.info('Starting BugBuddy thingy')
    commit = get(session, Commit, id=1809)

    go_to_commit(repository, commit, force=True)

    # run_all_tests(commit)

    for test_failure in commit.failed_test_results:
        predict_blame(test_failure)
    import pdb; pdb.set_trace()

    commit.summary()


def _watch(repository: Repository, commit_only: bool):
    '''
    Watches the repository's filesystem for changes and records the changes.
    It also notifies the user when there is an update in the test output.
    '''
    set_bug_buddy_branch(repository)
    logger.info('Starting BugBuddy watcher')
    event_handler = ChangeWatchdog(repository, commit_only)
    observer = Observer()
    observer.schedule(event_handler, repository.original_path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info('Shutting down BugBuddy watcher')
    observer.join()
