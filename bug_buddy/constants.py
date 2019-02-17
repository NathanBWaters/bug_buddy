
PYTHON_FILE_TYPE = 'PTYHON'
TMP_PATH = '/tmp/bug_buddy'
TEST_OUTPUT_FILE = 'BUG_BUDDY_TEST_OUTPUT_FILE'

# File types
FILE_TYPES = {
    PYTHON_FILE_TYPE: '.py'
}

TEST_OUTPUT_SUCCESS = 'success'
TEST_OUTPUT_FAILURE = 'failure'
TEST_OUTPUT_SKIPPED = 'skipped'

########################################################
#                 Synthetic Statements                 #
########################################################
# a statement that doesn't cause any tests to fail
BENIGN_STATEMENT = 'assert True'
# a statement that doesn't causes tests to fail
ERROR_STATEMENT = 'assert False'


########################################################
#                  Type of Changes                     #
########################################################
# the types of changes
DEVELOPER_CHANGE = 'developer'
SYNTHETIC_CHANGE = 'synthetic_alteration'
SYNTHETIC_FIXING_CHANGE = 'synthetic_fixing'
SYNTHETIC_RESET_CHANGE = 'synthetic_reset'
