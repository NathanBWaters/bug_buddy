'''
The constants for BugBuddy
'''
import os


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
# a developer change was created by snapshotting the changes created by a
# developer. In other words, this is 'real' data.
DEVELOPER_CHANGE = 'developer'

# the base synthetic change is the original change that creates unique
# synthetic changes.  We use the base synthetic alteration as an easy to
# retrieve location that contains all of the diffs.
BASE_SYNTHETIC_CHANGE = 'base_synthetic_alteration'

# a synthetic change was created using synthetic diffs of the base synthetic
# change
SYNTHETIC_CHANGE = 'synthetic_alteration'
SYNTHETIC_FIXING_CHANGE = 'synthetic_fixing'
SYNTHETIC_RESET_CHANGE = 'synthetic_reset'


########################################################
#                Environment Variables                 #
########################################################
MIRROR_ROOT = os.getenv('BUG_BUDDY_MIRROR_ROOT',
                        '/Users/NathanBWaters/code/.bug_buddy_mirror')
