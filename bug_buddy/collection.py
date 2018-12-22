'''
Reads from the output file a test run
'''
from junitparser import JUnitXml

from bug_buddy.constants import SUCCESS, FAILURE
from bug_buddy.db import get_or_create, create
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


def create_results_from_junit_xml(repository: Repository, test_run: TestRun):
    '''
    Gets results from a JUnitXML format file
    https://docs.pytest.org/en/latest/usage.html#creating-junitxml-format-files
    '''
    xml_output = JUnitXml.fromfile(test_run.output_file)

    for test_case in xml_output:
        test, _ = get_or_create(
            Test,
            repository=repository,
            name=test_case.name,
            file=test_case._elem.attrib.get('file'),
        )

        status = FAILURE if test_case.result else SUCCESS
        create(
            TestResult,
            test=test,
            test_run=test_run,
            status=status
        )
