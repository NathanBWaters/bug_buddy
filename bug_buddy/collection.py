'''
Reads from the output file a test run
'''
from junitparser import JUnitXml

from bug_buddy.constants import (TEST_OUTPUT_SUCCESS,
                                 TEST_OUTPUT_FAILURE,
                                 TEST_OUTPUT_SKIPPED)
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit


def record_test_results(repository: Repository, test_run: TestRun) -> dict:
    '''
    Analyze a repository for a given commit.  The results are saved to the
    database.

    @param repository: the repository to be analyzed
    @param run: the run to be analyzed
    '''
    print('Implement run_test')
    assert False


def create_results_from_junit_xml(output_file: str,
                                  repository: Repository,
                                  test_run: TestRun):
    '''
    Gets results from a JUnitXML format file
    https://docs.pytest.org/en/latest/usage.html#creating-junitxml-format-files
    '''
    session = Session.object_session(test_run)
    xml_output = JUnitXml.fromfile(output_file)

    for test_case in xml_output:
        test, _ = get_or_create(
            session,
            Test,
            repository=repository,
            name=test_case.name,
            file=test_case._elem.attrib.get('file'),
            classname=test_case._elem.attrib.get('classname'),
        )

        status = TEST_OUTPUT_FAILURE if test_case.result else TEST_OUTPUT_SUCCESS

        # if the test is skipped, do not keep it
        if hasattr(test_case, 'result') and hasattr(test_case.result, 'type'):
            if test_case.result.type == 'pytest.skip':
                status = TEST_OUTPUT_SKIPPED

        create(
            session,
            TestResult,
            test=test,
            test_run=test_run,
            status=status,
            time=test_case.time,
        )
