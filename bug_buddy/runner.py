'''
Commands for running tests and recording the output
'''
from datetime import datetime
import tempfile
import subprocess
import time

from bug_buddy.collection import create_results_from_junit_xml
from bug_buddy.db import session_manager, create, Session
from bug_buddy.git_utils import run_cmd
from bug_buddy.errors import BugBuddyError
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, TestRun, Commit


def run_test(repository: Repository, commit: Commit):
    '''
    Runs a repository's tests and records the results
    '''
    test_run = None

    if 'pytest' not in repository.test_commands:
        msg = 'BugBuddy has not implemented non-pytest testing yet'
        raise BugBuddyError(msg)

    try:
        temp_output = tempfile.NamedTemporaryFile(
            prefix="bugbuddy_test_output_",
            suffix=".xml")
        command = '{cmd} --junitxml={path}'.format(
            cmd=repository.test_commands,
            path=temp_output.name)

        start_timestamp = time.time()
        date = datetime.utcfromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        logger.info('Testing {repo_name} at commit "{commit_id}" at {date} '
                    'with command:\n{command}'
                    .format(repo_name=repository.name,
                            commit_id=commit.commit_id,
                            command=command,
                            date=date))

        process = subprocess.Popen(
            command,
            shell=True,
            cwd=repository.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        session = Session.object_session(repository)
        test_run = create(
            session,
            TestRun,
            commit=commit,
            start_timestamp=start_timestamp)

        # we were succesfully able to create a test run
        create_results_from_junit_xml(
            temp_output.name,
            repository,
            test_run)

        logger.info(test_run)

    finally:
        temp_output.close()

    return test_run


def library_is_testable(repository):
    '''
    Returns whether or not the library is testable.  It does this by running
    pytest --collect-only.  If there's anything in the stderr than we are
    assuming we have altered a method that is called during import of the
    library.  This is a huge limitation of bug_buddy.
    '''
    command = 'pytest --collect-only'
    stdout, stderr = run_cmd(repository, command)
    if stderr:
        return False

    return True
