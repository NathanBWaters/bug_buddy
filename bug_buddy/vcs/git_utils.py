'''
Git utility methods
'''
from git import Repo

from bug_buddy.errors import BugBuddyError


def get_repository_data_from_path(path: str):
    '''
    Gets repository data given a path
    '''
    repo = Repo(path)
    if repo.bare:
        msg = ('The path "{}" is either not a git project or is a bare git '
               'project'.format(path))
        BugBuddyError(msg)

    url = [url for url in repo.remote().urls if url is not None][0]
    name = url.split('/')[0]
    return url, name
