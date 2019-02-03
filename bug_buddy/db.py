#!/usr/bin/env python3
'''
Code for communicating with the bug_buddy database
'''
import ast
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from bug_buddy.errors import BugBuddyError
from bug_buddy.schema import Base, Repository, Function


# Create an engine that stores data in the local directory's
# sqlalchemy_example.db file.
engine = create_engine('sqlite:///bug_buddy.db')

# Create all tables in the engine. This is equivalent to "Create Table"
# statements in raw SQL.
Base.metadata.create_all(engine)
Session = sessionmaker()
Session.configure(bind=engine)


@contextmanager
def session_manager():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()


def get(session, sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    query = session.query(sql_class)
    class_instance = query.filter_by(**kwargs).first()
    return class_instance


def get_all(session, sql_class, **kwargs):
    '''
    Used to get all instances of a class that matche the kwargs parameters
    '''
    query = session.query(sql_class)
    class_instances = query.filter_by(**kwargs).all()
    return class_instances


def create(session, sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    new_class_instance = sql_class(**kwargs)
    session.add(new_class_instance)
    return new_class_instance


def get_or_create(session, sql_class, **kwargs):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    class_instance = get(session, sql_class, **kwargs)
    if class_instance:
        return class_instance, True

    class_instance = create(session, sql_class, **kwargs)
    return class_instance, False


def get_or_create_function(session,
                           repository: Repository,
                           node,
                           file_path: str):
    '''
    Given a function's ast node information, return the function from the
    database if it exists.  Otherwise, create the new function.
    '''
    matching_functions = get_all(
        session,
        Function,
        repository=repository,
        name=node.name)

    # if there are no matching functions, then create the new Function instance
    if not matching_functions:
        return create(session,
                      Function,
                      repository=repository,
                      node=node,
                      file_path=file_path)

    # the matching function is most likely the function with the line number
    # closest to the function's current line number in the source code.  This
    # is necessary for the case where a file has multiple functions with the
    # same name
    matching_function = min(
        matching_functions,
        key=lambda function: abs(function.line_number - node.lineno))

    # we must set the node information
    matching_function.node = node
    return matching_function


def delete(session, sql_instance):
    '''
    Used to get a single instance of a class that matches the kwargs parameters
    '''
    session.delete(sql_instance)


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
