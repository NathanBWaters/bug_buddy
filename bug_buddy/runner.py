'''
Commands for running tests and recording the output
'''
from datetime import datetime
import tempfile
import subprocess
import time

from bug_buddy.collection import create_results_from_junit_xml
from bug_buddy.db import create, Session
from bug_buddy.git_utils import run_cmd
from bug_buddy.logger import logger
from bug_buddy.schema import TestRun, Test, Commit, TestResult


def get_list_of_tests(commit: Commit) -> TestResult:
    '''
    Given a commit, it will return the list of available tests
    '''
    # TODO - this should actually check against the file system instead of
    # simply retreiving from the database
    return commit.repository.tests


def run_test(test_run: TestRun, test: Test) -> TestResult:
    '''
    Runs a specific test and returns the TestResult
    '''
    command = ('python -m pytest -vs {file}::{test_name}'
               .format(file=test.file, test_name=test.name))
    logger.info('Running individual test {} with command: "{}"'
                .format(test, command))

    _run_tests(test_run, command)

    # return the test output
    test_results = [test_result for test_result in test_run.test_results
                    if test_result.test.id == test.id]

    if not test_results:
        import pdb; pdb.set_trace(0)

    # if there are more than one, then return the one with the latest id
    test_results.sort(key=lambda test: test.id, reverse=False)

    return test_results[-1]


def run_all_tests(commit: Commit):
    '''
    Runs a repository's tests and records the results
    '''
    logger.info('Running the tests against commit: {}'.format(commit))
    test_run = None

    start_timestamp = time.time()
    date = datetime.utcfromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    logger.info('Testing {repo_name} at commit "{commit_id}" at {date} '
                'with command:\n{command}'
                .format(repo_name=commit.repository.name,
                        commit_id=commit.commit_id,
                        command=commit.repository.test_commands,
                        date=date))

    test_run = create(
        Session.object_session(commit),
        TestRun,
        commit=commit,
        start_timestamp=start_timestamp)

    # run all of the tests
    _run_tests(test_run, commit.repository.test_commands)

    return test_run


def _run_tests(test_run: TestRun, test_command: str):
    '''
    Runs a specific command and reads the output from the results file
    '''
    logger.info('Called runner::_run_tests with {test_run} and command "{cmd}"'
                .format(test_run=test_run, cmd=test_command))
    test_run = None

    try:
        temp_output = tempfile.NamedTemporaryFile(
            prefix="bugbuddy_test_output_",
            suffix=".xml")
        full_command = '{cmd} --junitxml={path}'.format(
            cmd=test_command,
            path=temp_output.name)

        process = subprocess.Popen(
            full_command,
            shell=True,
            cwd=test_run.commit.repository.path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        # we were succesfully able to create a test run
        create_results_from_junit_xml(
            temp_output.name,
            test_run.commit.repository,
            test_run)

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
    command = 'python -m pytest --collect-only'
    stdout, stderr = run_cmd(repository, command)

    # TODO - this is also really hacky
    if stderr or '=============== ERRORS ====================' in stdout:
        return False

    return True
