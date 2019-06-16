'''
This is the core class for performing operations on a code repository such as:
    1) Running a test
    2) Assigning blame
    3) Binary searching through a code repository to uncover bugs
'''
from bug_buddy.brain.predict_tests import predict_test_output
from bug_buddy.schema import Repository, Commit


class Brain(object):
    '''
    Class that makes decisions on which tests should be ran and which source
    code should be blamed

          _---~~(~~-_.
        _{        )   )
      ,   ) -~~- ( ,-' )_
     (  `-,_..`., )-- '_,)
    ( ` _)  (  -~( -_ `,  }
    (_-  _  ~_-~~~~`,  ,' )
      `~ -^(    __;-,((()))
            ~~~~ {_ -_(())
                   `{  }
                     { }
    '''
    def __init__(self, repository: Repository):
        '''
        Initialize the Brain
        '''
        self.repository = repository

    def set_commit(self, commit: Commit):
        '''
        Sets the commit to be operated on
        '''
        self.commit = commit

    def act(self):
        '''
        Performs an action.  For right now, this means it runs a test against
        a commit
        '''
        print('do')
        predict_test_output(self.commit)
