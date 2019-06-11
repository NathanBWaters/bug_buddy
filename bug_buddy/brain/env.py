'''
Reformulates a commit into a interactive environment
'''
from gym import Env

from bug_buddy.brain.utils import (
    commit_to_state,
    get_random_commits,
    TEST_STATUS_TO_ID_MAP,
    TEST_STATUS_LOC)
from bug_buddy.constants import TEST_OUTPUT_NOT_RUN
from bug_buddy.db import Session, create
from bug_buddy.logger import logger
from bug_buddy.schema import (
    Repository,
    TestRun,
    TestResult,
    Test)
from bug_buddy.runner import get_list_of_tests


class ChangeEnvironment(Env):
    '''
    An environment is the commit and interacting with that commit such as
    running tests against it and assigning blame
    '''
    def __init__(self, repository: Repository, synthetic_training=False):
        ''''
        Creates an environment from a commit
        '''
        # self.observation_space = spaces.Discrete(None)
        # self.action_space = spaces.Discrete(None)
        self.repository = repository
        self.synthetic_training = synthetic_training

        self.session = Session.object_session(repository)

    def set_commit(self, commit):
        '''
        Whenever we update the commit for the Environment, we need to set
        multiple different variables
        '''
        logger.info('Setting commit to {}'.format(commit))

        # If we are in synthetic training then show what the actual test
        # failures are at the beginning.
        if self.synthetic_training and False:
            logger.info('{commit} test failures: {test_failures}'
                        .format(commit=commit,
                                test_failures=commit.test_failures))

        self.session = Session.object_session(commit)
        self.commit = commit
        self.state = commit_to_state(
            commit,
            synthetic_training=self.synthetic_training)

        # All tests are linked with a corresponding TestRun instance.  When
        # running the tests against this state
        if not self.synthetic_training:
            self.test_run = create(self.session, TestRun, commit=self.commit)

        # list of available tests, sorted in order of their id.  It is sorted
        # by the Test.id
        self.all_tests = get_list_of_tests(self.commit)
        self.all_tests.sort(key=lambda test: test.id, reverse=False)

        # total number of rewards accumulated for this change
        self.total_rewards = 0

        # total number of newly passing or newly failing tests discovered for
        # this change
        self.num_newly_changed_results_found = 0

        # list of the tests that have already been ran for this commit
        self.tests_ran = []
        logger.info('Set commit')

    def step(self, action):
        '''
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (new_state, reward, done, info).

        # Arguments
            action (object): Agent chosen by the agent.  For example, the action
                             could be to run a test to see if it has newly
                             failed or newly passed status

        # Returns
            new_state (object): the new state of the current environment after
                                the action was taken
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further
                            step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for
                         debugging, and sometimes learning).
        '''
        # the action is the index of the test that the agent wants to run
        test = self.all_tests[action]

        # if we are in synthetic training, then we don't need to run the test.
        # We can simply retrieve from the database whether or not the test is
        # failing.
        if self.synthetic_training:
            test_result = self.get_synthetic_test_result(test)
            assert test_result

        else:
            raise NotImplementedError(
                'You have not implemented actually running a test')

        # determine the reward for running that particular test
        reward = self.get_reward(test_result)

        # update the state with the action
        self.update_state(test_result)

        #########################
        #      BOOKKEEPING      #
        #########################
        # store that we ran this test
        self.tests_ran.append(test)
        # store the total number of rewards we have received for this commit
        self.total_rewards += reward

        # if it was a newly failing or newly passing test result, then store
        # that information for bookeeping.
        if self.is_newly_changed_result(test_result):
            self.num_newly_changed_results_found += 1

        # we can store extra debugging information here
        info_dict = {}

        # hiding this for now
        if False:
            logger.info('Commit #{commit} | '
                        '{test} | '
                        '{previous_status} -> {new_status} | '
                        'R: {reward} | '
                        'T: {total} | '
                        '{num_tests_ran} / {total_tests}'
                        .format(commit=self.commit.id,
                                test=test.name,
                                previous_status=self.get_previous_status(test),
                                new_status=test_result.status,
                                reward=reward,
                                total=self.total_rewards,
                                num_tests_ran=len(self.tests_ran),
                                total_tests=len(self.all_tests)))

        if self.done:
            logger.info(
                'Completed commit #{commit} | '
                'Total reward: {total} / {total_possible} | '
                '{num_tests_ran} / {total_tests}'
                .format(commit=self.commit.id,
                        total=self.total_rewards,
                        total_possible=self.num_newly_changed_results_found,
                        num_tests_ran=len(self.tests_ran),
                        total_tests=len(self.all_tests)))

        return self.state, reward, self.done, info_dict

    def get_reward(self, test_result: TestResult):
        '''
        Returns the reward for the action.  The goal is to reward the agent so
        that it runs in the tests in the following order:

            1) Run all newly passing or failing tests
            2) Run the rest of the tests

        The rewards are calculated in the
        following way:

            1) Discover newly failing or newly passing test == 1 - percentage
               of tests already ran that were not newly passing/failing.

            2) No change in test status == 0

            3) Ran a test that was already ran in the test run == -10
        '''
        # if the commit has already ran the test, then we should heavily
        # punish the agent for being wasteful.  What is currently stored in the
        # state for the test should ideally be TEST_OUTPUT_NOT_RUN.
        if (self.get_current_status(test_result.test) != TEST_OUTPUT_NOT_RUN):
            return -10

        # If the status is different from the previous commit's status and we
        # have already ruled out skips, then we should positively reward the
        # agent while also subtracting how many non-newly failing/passing tests
        # have already been ran.
        elif (self.is_newly_changed_result(test_result)):
            # the reward is based on how many it has succesfully discovered
            # so far plus getting it right this time.  The +1 is due to the fact
            # that it just discovered one
            return ((self.num_newly_changed_results_found + 1) /
                    (float(len(self.tests_ran) + 1)))

        # If the status is the same in this commit as it was in the previous
        # commit, simply return 0.
        return 0

    def is_newly_changed_result(self, test_result: TestResult) -> str:
        '''
        Returns if the status for a test changed from the previous commit to the
        current commit
        '''
        return not(test_result.status ==
                   self.get_previous_status(test_result.test))

    def get_current_status(self, test: Test) -> str:
        '''
        Returns the current status in the state vector for a test
        '''
        # the location of the test in the state tensor
        state_test_id = self.get_state_test_id(test)
        status_id = int(self.state[0][0][state_test_id][TEST_STATUS_LOC])
        status_lookup = {value: key for key, value in
                         TEST_STATUS_TO_ID_MAP.items()}
        return status_lookup[status_id]

    def get_previous_status(self, test: Test) -> str:
        '''
        Returns the previous status in the state vector for a test
        '''
        # the location of the test in the state tensor
        state_test_id = self.get_state_test_id(test)
        status_id = int(self.state[1][0][state_test_id][TEST_STATUS_LOC])
        status_lookup = {value: key for key, value in
                         TEST_STATUS_TO_ID_MAP.items()}
        return status_lookup[status_id]

    def update_state(self, test_result: TestResult):
        '''
        Given the output of a test, update the state
        '''
        # logger.info('Updating state for {test} with status {status}'
        #             .format(test=test_result.test,
        #                     status=test_result.status))
        status_id = TEST_STATUS_TO_ID_MAP[test_result.status]
        test_index = self.get_state_test_id(test_result.test)
        self.state[0][:, test_index, TEST_STATUS_LOC] = status_id

    def get_state_test_id(self, test):
        '''
        Given a test, it will return the state id for that test.  The state id
        represents the corresponding location in the state tensor for the test.
        It is ordered by the Test.id.
        '''
        try:
            return self.all_tests.index(test)
        except ValueError:
            import pdb; pdb.set_trace()

    @property
    def done(self):
        '''
        Whether or not this episode is complete.  An episode is complete for a
        given commit if all the tests are ran against the commit.  Note that
        a test can be ran multiple times by the agent, so we should not count
        duplicates.
        '''
        return len(list(set(self.tests_ran))) == len(self.all_tests)

    def get_synthetic_test_result(self, test_to_run):
        '''
        Given a test to run, look and see for the synthetic test's history what
        the result is
        '''
        assert len(self.commit.test_runs) == 1
        test_run = self.commit.test_runs[0]
        for test_result in test_run.test_results:
            if test_result.test.id == test_to_run.id:
                return test_result

    def reset(self):
        '''
        Resets the state of the environment and returns an initial state.

        # Returns
            state (object): The initial state of the space. Initial reward is
                            assumed to be 0.
        '''
        if self.synthetic_training:
            # retrieve a random synthetic commit and set the environment to
            # that commit
            commit = get_random_commits(
                self.repository, num_commits=1, synthetic=True)
            # re-initialize the environment with the new commit
            self.set_commit(commit)

            return self.state
        else:
            raise NotImplementedError(
                'In _reset you need to set synthetic_training to True since'
                'non-synthetic training steps are not implemented')

    def render(self, mode='human', close=False):
        '''
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        '''
        raise NotImplementedError()
