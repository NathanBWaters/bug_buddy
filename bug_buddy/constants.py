
PYTHON_FILE_TYPE = 'PTYHON'
TMP_PATH = '/tmp/bug_buddy'
TEST_OUTPUT_FILE = 'BUG_BUDDY_TEST_OUTPUT_FILE'

# File types
FILE_TYPES = {
    PYTHON_FILE_TYPE: '.py'
}

SUCCESS = 'success'
FAILURE = 'failure'


########################################################
#                       Diff Types                     #
########################################################
DIFF_ADDITION = 'addition'
DIFF_SUBTRACTION = 'subtraction'


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
