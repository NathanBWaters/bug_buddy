#!/usr/bin/env python3
'''
Object representation a Diff.  Pretty simple, only works with additions
currently
'''
from sqlalchemy import Column, ForeignKey, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from typing import List

from bug_buddy.schema.base import Base
from bug_buddy.schema.commit import Commit, Diff


class DiffCommitLink(Base):
    '''
    Links a DiffCommitLink with a Commit
    '''
    __tablename__ = 'diff_commit_link'

    id = Column(Integer, primary_key=True)

    commit_id = Column(Integer, ForeignKey('commit.id'))
    commit = relationship('Commit', back_populates='diffs')

    diff_id = Column(Integer, ForeignKey('diff.id'))
    diff = relationship('Diff', back_populates='commits')

    def __init__(self,
                 commit: Commit,
                 diff: Diff):
        '''
        Creates a new DiffCommitLink instance.
        '''
        self.commit = commit
        self.diff = diff

    def __repr__(self):
        '''
        Converts the DiffCommitLink into a string
        '''
        return ('<DiffCommitLink {diff} | {commit} />'
                .format(diff=self.diff,
                        commit=self.commit))
