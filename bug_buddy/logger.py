'''
bug_buddy's logger
'''

import logging
logger = logging.getLogger('bug_buddy')
logger.setLevel(level=logging.DEBUG)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter(
    fmt='%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
fh.setFormatter(fh_formatter)
logger.addHandler(fh)
