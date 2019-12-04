import enum
from sc2 import Race
from sc2.player import Bot
from sc2rl.environments.EnvironmentBase import SC2EnvironmentBase
from sc2rl.environments.SC2BotAI import SimpleSC2BotAI
from sc2rl.utils.sc2_utils import get_random_action

VERY_LARGE_NUMBER = 1e10


class Status(enum.Enum):
    RUNNING = 0
    END = 1


class MicroTestEnvironment(SC2EnvironmentBase):

    def __init__(self, map_name, reward_func, state_proc_func, realtime=False, max_steps=25000,
                 winning_ratio_gamma=0.1, frame_skip_rate=1):
        """
        :param map_name:
        :param reward_func:
        :param state_proc_func:
        :param realtime:
        :param max_steps: (int) max step integrations. 50000 is tested.
        """

        allies = Bot(Race.Terran, SimpleSC2BotAI())
        super(MicroTestEnvironment, self).__init__(map_name=map_name,
                                                   allies=allies,
                                                   realtime=realtime,
                                                   frame_skip_rate=frame_skip_rate)
        self.max_steps = max_steps
        self._step_count = 0
        self.status = Status.RUNNING

        self.reward_func = reward_func
        self.state_proc_func = state_proc_func
        self.prev_health = VERY_LARGE_NUMBER
        self.curr_health = VERY_LARGE_NUMBER

        self.winning_ratio = 0.0
        self.winning_ratio_gamma = winning_ratio_gamma

    @property
    def step_count(self):
        return self._step_count

    @step_count.setter
    def step_count(self, s_count):
        self._step_count = s_count
        if self.step_count >= self.max_steps:
            self.status = Status.END

    def reset(self):
        sc2_game_state = self._reset()
        self.step_count = 0
        self.status = Status.RUNNING
        return self.state_proc_func(sc2_game_state)

    def observe(self):
        sc2_game_state = self._observe()
        return self.state_proc_func(sc2_game_state)

    def _check_done(self, sc2_game_state):
        num_allies = len(sc2_game_state.units.owned)
        num_enemies = len(sc2_game_state.units.enemy)
        cur_health = 0
        for u in sc2_game_state.units:
            cur_health += u.health
        self.curr_health = cur_health

        done_increase = num_allies == 0 or num_enemies == 0

        if self.prev_health < self.curr_health:
            done_zero_units = True
        else:
            done_zero_units = False
        self.prev_health = self.curr_health

        return done_increase or done_zero_units

    def step(self, action=None):
        self.step_count = self.step_count + 1
        sc2_cur_state = self._observe()
        if action is None:
            action = get_random_action(sc2_cur_state)

        sc2_next_state, _ = self._step(action_args=action)

        # additional routine for checking done!
        # Done checking behaviour of the variants of 'MicroTest' are different from the standard checking done routine.
        done = self._check_done(sc2_next_state)

        cur_state = self.state_proc_func(sc2_cur_state)
        next_state = self.state_proc_func(sc2_next_state)
        reward = self.reward_func(cur_state, next_state, done)

        if done:  # Burn few remaining frames
            win = int(len(sc2_next_state.units.owned) >= len(sc2_next_state.units.enemy))

            self.burn_last_frames()
            if self.status == Status.END:
                _ = self.reset()

            gamma = self.winning_ratio_gamma
            self.winning_ratio = gamma * win + (1 - gamma) * self.winning_ratio

        return next_state, reward, done

    def burn_last_frames(self):
        while True:
            self.step_count = self.step_count + 1
            sc2_cur_state = self._observe()
            done = self._check_done(sc2_cur_state)
            _, _ = self._step(action_args=None)

            if not done:
                break
