#!/usr/bin/env python3
import argparse
from collections import defaultdict
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from bug_buddy.db import get_all, session_manager, delete
from bug_buddy.logger import logger
from bug_buddy.schema import TestRun


if __name__ == '__main__':
    with session_manager() as session:
        test_runs = get_all(session, TestRun)
        for test_run in test_runs:
            test_result_dict = defaultdict(list)
            for test_result in test_run.test_results:
                unique_id = '{}/{}/{}'.format(test_result.test.name,
                                              test_result.test.file,
                                              test_result.test.classname)
                test_result_dict[unique_id].append(test_result)

            for test_name in test_result_dict.keys():
                if len(test_result_dict[test_name]) > 1:
                    duplicates = test_result_dict[test_name]
                    earliest = min(duplicates, key=lambda test_result: test_result.id)

                    for dupe in duplicates:
                        if (len(duplicates) > 2 or
                                abs(duplicates[1].id - duplicates[0].id) != 1):
                            import pdb; pdb.set_trace()
                        if dupe is not earliest:
                            logger.info('deleting duplicate {}'.format(dupe))
                            delete(session, dupe)
