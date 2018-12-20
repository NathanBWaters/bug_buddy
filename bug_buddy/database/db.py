#!/usr/bin/env python3
'''
Code for communicating with the bug_buddy database
'''
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bug_buddy.errors import BugBuddyError
from bug_buddy.schema import Base, Repository
from bug_buddy.vcs.git_utils import get_name_from_url


# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine('sqlite:///bug_buddy.db')

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)
Session = sessionmaker()
Session.configure(bind=engine)


def get(sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    session = Session()
    query = session.query(sql_class)
    class_instance = query.filter_by(**kwargs).first()
    session.commit()
    return class_instance


def create(sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    session = Session()
    new_class_instance = sql_class(**kwargs)
    session.add(new_class_instance)
    session.commit()
    return new_class_instance


def get_or_create(sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    class_instance = get(sql_class, **kwargs)
    if class_instance:
        return class_instance, True

    class_instance = create(sql_class, **kwargs)
    return class_instance, False


def get_or_create_repository(name: str=None,
                             url: str=None,
                             path: str=None,
                             test_commands: str=None):
    '''
    Returns a repository, either by retrieving it from the database or creating
    it and then returning it.  This method is complicated because there are
    various states a repository could be in:

        1) Not created locally or in db
        2) Created locally but not in db
        3) Not created locally but stored in db
        4) Created locally and in db

    '''
    # if the path is not specified, then assume it is not created and that we
    # need to create a local copy
    if not path:
        # if not the name or url or path were specified, then we don't have enough
        # data to get the repository
        if not name and not url:
            msg = ('You did not specify a name, url or path when trying to get '
                   'a repository.  Unable to create a local copy or determine '
                   'which repository you are referring to')
            raise BugBuddyError(msg)

        # if the user specifies only a name, that's enough if it's in the database.
        # We are assuming that the project does not exist locally, so we will create
        # it
        if name and not url:
            repository = get(Repository, name=name)

            if not repository:
                msg = ('The name "{}" does not exist in the database for a '
                       'repository.'
                       .format(name))
                raise BugBuddyError(msg)

            repository.create_local_copy()
            return repository

        if url:
            repository = get(Repository, url=url)
            if not repository:
                msg = ('The url "{}" does not exist in the database for a '
                       'repository.'
                       .format(url))
                raise BugBuddyError(msg)

            repository.create_local_copy()
            return repository

    # we have a path, see if it's located in the database.
    else:
        pass
