'''
CLI output for visualizing the results of the ChangeWatchdog
'''
from blessings import Terminal
import emoji

from bug_buddy.schema import Commit


class Scorecard(Terminal):
    '''
    CLI output for visualizing the results of the ChangeWatchdog
    '''

    def __init__(self):
        '''
        Create a ScoreCard instance
        '''
        super().__init__()
        self.num_correct_total = 0
        self.num_incorrect_total = 0

        self.commit = None

    @property
    def score(self):
        '''
        The number of times it has correctly guessed the output of a test
        '''
        if not self.num_correct_total + self.num_incorrect_total:
            return 0.0

        return self.num_correct_total / (
            float(self.num_correct_total + self.num_incorrect_total))

    def render(self, commit: Commit=None):
        '''
        Renders the current state at a commit
        '''
        with self.fullscreen():
            self.clear()
            print("")
            print('    🐞 ' + self.underline("BugBuddy"))

            with self.location(x=self.width - 30, y=1):
                print('Score: {}'.format(self.score))

            print('')
            print('-' * self.width)
            print('')
            if commit:
                print('...')
