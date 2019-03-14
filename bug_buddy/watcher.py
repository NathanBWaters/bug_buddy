'''
The watcher records a developer's changes
'''
import sys
import time
import logging
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

from bug_buddy.schema import Repository
from bug_buddy.logger import logger


class Watchdog(FileSystemEventHandler):
    '''
    Is notified every time an event occurs on the fileystem
    '''
    def on_any_event(self, event):
        '''
        Catches all events
        '''
        print('event: ', event)


def watch(repository: Repository):
    '''
    Watches the repository's filesystem for changes and records the changes.
    It also notifies the user when there is an update in the test output.
    '''
    event_handler = Watchdog()
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


