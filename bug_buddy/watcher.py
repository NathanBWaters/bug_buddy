'''
The watcher records a developer's changes
'''
import os
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, PatternMatchingEventHandler

from bug_buddy.constants import MIRROR_ROOT
from bug_buddy.db import get, session_manager
from bug_buddy.git_utils import run_cmd, is_repo_clean, set_bug_buddy_branch
from bug_buddy.logger import logger
from bug_buddy.schema import Repository
from bug_buddy.snapshot import snapshot
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

    def on_any_event(self, event):
        '''
        Catches all events
        '''
        if '/.' in event.src_path:
            return

        print('{} event: {}'.format(self.repository.name, event))

        updated_file = os.path.relpath(event.src_path,
                                       self.repository.original_path)
        if (not updated_file or updated_file in self.repository.ignored_files or
                not updated_file.endswith('.py')):
            logger.info('Ignoring update to {}'.format(updated_file))
            return

        # we have to recreate the repository in this thread for Sqlite
        with session_manager() as session:
            repository = get(session, Repository, id=self.repository.id)
            logger.info('Syncing updates')
            # Copy the change over to the mirror repository
            sync_mirror_repo(repository)

            if not is_repo_clean(self.repository):
                # make sure the repository is on the bug_buddy branch
                commit = snapshot(repository, commit_only=self.commit_only)
                print('Completed snapshot of {}'.format(commit))

                # run the tests in a appropriate order
                session.commit()
            else:
                logger.info('Nothing was changed')


def watch(repository: Repository, commit_only: bool):
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
