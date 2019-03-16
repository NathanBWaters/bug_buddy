'''
The watcher records a developer's changes
'''
import os
import time
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

from bug_buddy.constants import MIRROR_ROOT
from bug_buddy.git_utils import run_cmd, is_repo_clean
from bug_buddy.logger import logger
from bug_buddy.schema import Repository


class Watchdog(FileSystemEventHandler):
    '''
    Is notified every time an event occurs on the fileystem
    '''
    def __init__(self, repository):
        '''
        Create a Watchdog instance
        '''
        super().__init__()
        self.repository = repository

    def on_any_event(self, event):
        '''
        Catches all events
        '''
        print('{} event: {}'.format(self.repository.name, event))

        # make sure there is an actual change recognized by git
        if not is_repo_clean(self.repository):
            print('Updating the mirror repository')
            # Copy the change over to the mirror repository
            update_mirror_repo(self.repository)

            # make sure the repository is on the bug_buddy branch

            # create a snapshot of the changes


def update_mirror_repo(repository: Repository):
    '''
    Updates the mirror repository to match the code base the developer is
    working on
    '''
    command = ('rsync -a {source} {destination}'
               .format(source=repository.path,
                       destination=MIRROR_ROOT))

    if not os.path.exists(MIRROR_ROOT):
        os.makedirs(MIRROR_ROOT)

    run_cmd(repository, command, log=True)


def watch(repository: Repository):
    '''
    Watches the repository's filesystem for changes and records the changes.
    It also notifies the user when there is an update in the test output.
    '''
    event_handler = Watchdog(repository)
    observer = Observer()
    observer.schedule(event_handler, repository.path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logger.info('Shutting down BugBuddy watcher')
    observer.join()


