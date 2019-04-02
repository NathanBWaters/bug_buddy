'''
Reads from the output file a test run
'''
from junitparser import JUnitXml

from bug_buddy.constants import (TEST_OUTPUT_SUCCESS,
                                 TEST_OUTPUT_FAILURE,
                                 TEST_OUTPUT_SKIPPED)
from bug_buddy.db import Session, get_or_create, create, session_manager
from bug_buddy.logger import logger
from bug_buddy.schema import Repository, TestResult, Test, TestRun, Commit


def create_results_from_junit_xml(output_file: str,
                                  repository: Repository,
                                  test_run: TestRun):
    '''
    Gets results from a JUnitXML format file
    https://docs.pytest.org/en/latest/usage.html#creating-junitxml-format-files
    '''
    logger.info('Reading info from {}'.format(output_file))
    try:
        session = Session.object_session(test_run)
        xml_output = JUnitXml.fromfile(output_file)

        test_names = []

        expected_new = False

        for test_case in xml_output:
            # There can seemingly be duplicate test outputs for a test if both
            # the test and the test's teardown step both fail.  So we will ignore
            # the second test output
            unique_id = '{}/{}/{}'.format(test_case.name,
                                          test_case._elem.attrib.get('file'),
                                          test_case._elem.attrib.get('classname'))
            if unique_id in test_names:
                logger.error('There was a duplicate test output for test: {}'
                             .format(test_case.name))
                continue

            test_names.append(unique_id)

            test, is_new = get_or_create(
                session,
                Test,
                repository=repository,
                name=test_case.name,
                file=test_case._elem.attrib.get('file'),
                classname=test_case._elem.attrib.get('classname'),
            )

            if is_new and not expected_new:
                import pdb; pdb.set_trace()
                logger.error('Did you expect to create a new test?')
                expected_new = True

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

    except Exception as e:
        import pdb; pdb.set_trace()
        logger.info('Hit error when reading from junit xml: {}'.format(e))
