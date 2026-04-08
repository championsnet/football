# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full 11 vs 11 cooperative multi-agent PPO training.

Architecture overview
---------------------
Each of the 11 left-team players is treated as an independent agent that
shares a single PPO policy (parameter sharing / centralised-training
decentralised-execution).

  N games × 11 players per game  →  11·N "virtual" VecEnv slots

All 11·N slots are batched together by SB3's PPO, so a single MLP policy
receives every player's 115-dim observation and outputs their action.

Usage
-----
  # Train against easy built-in AI (default)
  python3 -m gfootball.examples.run_11v11

  # Harder opponent
  python3 -m gfootball.examples.run_11v11 --level=11_vs_11_hard_stochastic

  # Resume from a checkpoint
  python3 -m gfootball.examples.run_11v11 --load_path=ppo_logs_11v11/checkpoints/ppo_model_500000_steps
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

from absl import app
from absl import flags

from gfootball.env import config as env_config
from gfootball.env import football_env as raw_football_env
from gfootball.env import wrappers as legacy_wrappers
import numpy as np

try:
  import gymnasium as gym
  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import CheckpointCallback
  from stable_baselines3.common.vec_env import VecEnv, VecMonitor
except ImportError as exc:
  raise ImportError(
      'Training requires `stable-baselines3` and `gymnasium`. '
      'Install with: uv pip install stable-baselines3 gymnasium shimmy'
  ) from exc


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string('level', '11_vs_11_easy_stochastic',
                    'Football scenario. Must be an 11v11 scenario name.')
flags.DEFINE_enum('reward_experiment', 'scoring,checkpoints',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward shaping: scoring only, or scoring + dense checkpoints.')
flags.DEFINE_integer('num_timesteps', int(50_000_000),
                     'Total environment steps (across all virtual envs).')
flags.DEFINE_integer('num_games', 4,
                     'Number of parallel 11v11 games. '
                     'Total VecEnv width = num_games × 11.')
flags.DEFINE_integer('nsteps', 256,
                     'Per-player rollout steps per PPO update. '
                     'Rollout buffer = num_games × 11 × nsteps.')
flags.DEFINE_integer('noptepochs', 4, 'PPO optimisation epochs per update.')
flags.DEFINE_integer('nminibatches', 8, 'Number of minibatches per update.')
flags.DEFINE_integer('save_interval', 100,
                     'Save a checkpoint every this many PPO updates.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate.')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coefficient.')
flags.DEFINE_float('gamma', 0.993, 'Discount factor.')
flags.DEFINE_float('cliprange', 0.27, 'PPO clip range.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm.')
flags.DEFINE_bool('render', False,
                  'Render the first game to screen (slows training).')
flags.DEFINE_bool('dump_full_episodes', False,
                  'Dump a trace after every episode of game 0.')
flags.DEFINE_bool('dump_scores', False,
                  'Dump sampled traces after scoring events in game 0.')
flags.DEFINE_string('load_path', None,
                    'Resume training from this SB3 model checkpoint.')
flags.DEFINE_string('logdir', 'ppo_logs_11v11',
                    'Directory for monitors, checkpoints, and final model.')
flags.DEFINE_string('device', 'auto',
                    'PyTorch device: `auto`, `cpu`, or `cuda`.')


# ---------------------------------------------------------------------------
# Per-game multi-player environment
# ---------------------------------------------------------------------------

_NUM_PLAYERS = 11
_OBS_DIM = 115
_NUM_ACTIONS = 19


class MultiPlayerFootballEnv:
  """One physical 11v11 game with all 11 left players controlled.

  `reset()` returns numpy array of shape (11, 115).
  `queue_actions(actions)` stores the 11-element action array.
  `step_queued()` executes the queued actions and returns
    (obs (11, 115), rewards (11,), done bool, info dict).

  The right team is controlled by the built-in AI (difficulty set by the
  scenario; start with `11_vs_11_easy_stochastic`).
  """

  def __init__(self, rank, logdir, train_config):
    self._rank = rank
    self._reward_experiment = train_config['reward_experiment']

    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1
    self._collected_checkpoints = [0] * _NUM_PLAYERS

    config_values = {
        'dump_full_episodes': train_config['dump_full_episodes'] and rank == 0,
        'dump_scores': train_config['dump_scores'] and rank == 0,
        # Control all 11 left players; right team is built-in AI.
        'players': ['agent:left_players=11,right_players=0'],
        'level': train_config['level'],
        'tracesdir': str(logdir),
        'write_video': False,
    }
    self._config = env_config.Config(config_values)
    self._env = raw_football_env.FootballEnv(self._config)
    if train_config['render'] and rank == 0:
      self._env.render()

    self._queued_actions = None

  # ------------------------------------------------------------------
  # Public interface
  # ------------------------------------------------------------------

  def reset(self):
    """Reset the game and return initial observations (11, 115)."""
    self._collected_checkpoints = [0] * _NUM_PLAYERS
    raw_obs = self._env.reset()
    return self._convert_obs(raw_obs)

  def queue_actions(self, actions):
    """Buffer the 11 discrete actions to be executed on next step_queued."""
    self._queued_actions = np.asarray(actions, dtype=np.int32)

  def step_queued(self):
    """Execute buffered actions. Returns (obs, rewards, done, info)."""
    assert self._queued_actions is not None, 'Call queue_actions() first.'
    raw_obs, raw_rewards, done, info = self._env.step(
        list(self._queued_actions))
    self._queued_actions = None

    obs = self._convert_obs(raw_obs)  # (11, 115)
    rewards = np.array(raw_rewards, dtype=np.float32)  # (11,)
    if 'checkpoints' in self._reward_experiment:
      rewards = self._apply_checkpoint_rewards(rewards, raw_obs)
    return obs, rewards, bool(done), info

  def close(self):
    self._env.close()

  # ------------------------------------------------------------------
  # Internals
  # ------------------------------------------------------------------

  def _convert_obs(self, raw_obs):
    """Convert a list of 11 raw observation dicts to (11, 115) float32."""
    arr = legacy_wrappers.Simple115StateWrapper.convert_observation(
        raw_obs, True)
    return np.array(arr, dtype=np.float32)

  def _apply_checkpoint_rewards(self, rewards, raw_obs):
    """Dense reward shaping: award 0.1 per checkpoint crossed toward goal."""
    for i in range(_NUM_PLAYERS):
      obs_i = raw_obs[i]
      r = float(rewards[i])
      if r == 1.0:
        r += self._checkpoint_reward * (
            self._num_checkpoints - self._collected_checkpoints[i])
        self._collected_checkpoints[i] = self._num_checkpoints
      elif (obs_i.get('ball_owned_team') == 0 and
            obs_i.get('ball_owned_player') == obs_i.get('active')):
        ball = obs_i.get('ball', [0, 0, 0])
        distance = ((ball[0] - 1) ** 2 + ball[1] ** 2) ** 0.5
        while self._collected_checkpoints[i] < self._num_checkpoints:
          if self._num_checkpoints == 1:
            threshold = 0.99 - 0.8
          else:
            threshold = (
                0.99 - 0.8 / (self._num_checkpoints - 1) *
                self._collected_checkpoints[i])
          if distance > threshold:
            break
          r += self._checkpoint_reward
          self._collected_checkpoints[i] += 1
      rewards[i] = r
    return rewards


# ---------------------------------------------------------------------------
# Custom VecEnv: N games × 11 players = 11·N VecEnv slots
# ---------------------------------------------------------------------------

class FootballMultiAgentVecEnv(VecEnv):
  """Stable-Baselines3 VecEnv that exposes one slot per player per game.

  Slots [i*11 .. i*11+10] all belong to game i.  When game i's episode
  finishes, all 11 of its slots are auto-reset (SB3 convention: terminal
  observation is stored in info['terminal_observation']).

  A single shared PPO policy sees every slot's 115-dim observation and
  produces its discrete action.  This implements full parameter sharing
  for cooperative 11v11 play.
  """

  def __init__(self, num_games, train_config, logdir):
    obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(_OBS_DIM,), dtype=np.float32)
    act_space = gym.spaces.Discrete(_NUM_ACTIONS)
    super().__init__(num_games * _NUM_PLAYERS, obs_space, act_space)

    self._num_games = num_games
    self._games = [
        MultiPlayerFootballEnv(i, logdir, train_config)
        for i in range(num_games)
    ]
    self._actions = None

  # ------------------------------------------------------------------
  # VecEnv interface
  # ------------------------------------------------------------------

  def reset(self):
    obs_parts = [game.reset() for game in self._games]
    return np.concatenate(obs_parts, axis=0)  # (11·N, 115)

  def step_async(self, actions):
    self._actions = np.asarray(actions, dtype=np.int32)

  def step_wait(self):
    all_obs = []
    all_rewards = []
    all_dones = []
    all_infos = []

    for i, game in enumerate(self._games):
      game_actions = self._actions[i * _NUM_PLAYERS:(i + 1) * _NUM_PLAYERS]
      game.queue_actions(game_actions)
      obs, rewards, done, info = game.step_queued()

      if done:
        terminal_obs = obs.copy()  # (11, 115)
        obs = game.reset()         # (11, 115) — fresh episode
        for j in range(_NUM_PLAYERS):
          all_infos.append(dict(info, terminal_observation=terminal_obs[j]))
      else:
        base_info = dict(info) if isinstance(info, dict) else {}
        for j in range(_NUM_PLAYERS):
          all_infos.append(base_info)

      all_obs.append(obs)
      all_rewards.append(rewards)
      all_dones.extend([done] * _NUM_PLAYERS)

    obs_arr = np.concatenate(all_obs, axis=0)      # (11·N, 115)
    rew_arr = np.concatenate(all_rewards, axis=0)  # (11·N,)
    done_arr = np.array(all_dones, dtype=bool)     # (11·N,)
    return obs_arr, rew_arr, done_arr, all_infos

  def close(self):
    for game in self._games:
      game.close()

  # ------------------------------------------------------------------
  # VecEnv abstract methods (minimal stubs required by SB3)
  # ------------------------------------------------------------------

  def get_attr(self, attr_name, indices=None):
    indices = self._get_indices(indices)
    return [getattr(self._games[i // _NUM_PLAYERS], attr_name, None)
            for i in indices]

  def set_attr(self, attr_name, value, indices=None):
    indices = self._get_indices(indices)
    for i in indices:
      setattr(self._games[i // _NUM_PLAYERS], attr_name, value)

  def env_method(self, method_name, *method_args, indices=None,
                 **method_kwargs):
    indices = self._get_indices(indices)
    return [
        getattr(self._games[i // _NUM_PLAYERS], method_name)(
            *method_args, **method_kwargs)
        for i in indices
    ]

  def seed(self, seed=None):
    return [None] * self.num_envs

  def render(self, mode='human'):
    pass

  def get_images(self):
    return [None] * self.num_envs

  def env_is_wrapped(self, wrapper_class, indices=None):
    return [False] * len(self._get_indices(indices))

  def _get_indices(self, indices):
    if indices is None:
      return list(range(self.num_envs))
    return list(indices)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(_):
  """Train a shared PPO policy over all 11 left-team players."""
  # Build a plain dict so it can be passed around without pickling issues.
  train_config = {
      'level': FLAGS.level,
      'reward_experiment': FLAGS.reward_experiment,
      'render': FLAGS.render,
      'dump_full_episodes': FLAGS.dump_full_episodes,
      'dump_scores': FLAGS.dump_scores,
      'num_games': FLAGS.num_games,
  }

  total_virtual_envs = FLAGS.num_games * _NUM_PLAYERS
  rollout_size = total_virtual_envs * FLAGS.nsteps
  if rollout_size % FLAGS.nminibatches != 0:
    raise ValueError(
        'num_games × 11 × nsteps ({}) must be divisible by nminibatches ({}). '
        'Adjust --nminibatches or --nsteps.'.format(
            rollout_size, FLAGS.nminibatches))

  logdir = Path(FLAGS.logdir).resolve()
  logdir.mkdir(parents=True, exist_ok=True)

  print('--- 11v11 multi-agent PPO ---')
  print('  Level            :', FLAGS.level)
  print('  Parallel games   :', FLAGS.num_games)
  print('  Virtual VecEnv   :', total_virtual_envs)
  print('  Rollout per update:', rollout_size)
  print('  Mini-batch size  :', rollout_size // FLAGS.nminibatches)
  print('  Log dir          :', logdir)
  print('-----------------------------')

  vec_env = FootballMultiAgentVecEnv(FLAGS.num_games, train_config, logdir)
  vec_env = VecMonitor(vec_env, filename=str(logdir / 'vec_monitor.csv'))

  batch_size = rollout_size // FLAGS.nminibatches
  checkpoint_dir = logdir / 'checkpoints'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_callback = CheckpointCallback(
      save_freq=max(FLAGS.save_interval * FLAGS.nsteps, 1),
      save_path=str(checkpoint_dir),
      name_prefix='ppo_11v11')

  # Shared MLP policy: each player's 115-dim observation → action.
  # Two hidden layers of 512 and 256 units handle the complexity of
  # a full 11v11 game while remaining fast to train.
  policy_kwargs = dict(net_arch=[512, 256])

  if FLAGS.load_path:
    model = PPO.load(
        FLAGS.load_path,
        env=vec_env,
        device=FLAGS.device,
        print_system_info=True,
    )
  else:
    model = PPO(
        'MlpPolicy',
        vec_env,
        verbose=1,
        seed=FLAGS.seed,
        learning_rate=FLAGS.lr,
        n_steps=FLAGS.nsteps,
        batch_size=batch_size,
        n_epochs=FLAGS.noptepochs,
        gamma=FLAGS.gamma,
        ent_coef=FLAGS.ent_coef,
        clip_range=FLAGS.cliprange,
        max_grad_norm=FLAGS.max_grad_norm,
        tensorboard_log=None,
        device=FLAGS.device,
        policy_kwargs=policy_kwargs,
    )

  try:
    model.learn(
        total_timesteps=FLAGS.num_timesteps,
        callback=checkpoint_callback,
    )
    model.save(str(logdir / 'final_model'))
    print('Saved final model to', logdir / 'final_model')
  finally:
    vec_env.close()


if __name__ == '__main__':
  app.run(train)
