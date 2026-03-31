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

"""Runs football_env on a modern PPO stack.

This replaces the legacy TF1/OpenAI-Baselines PPO2 example with an equivalent
Stable-Baselines3 PPO setup that works on modern Python versions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from pathlib import Path

from absl import app
from absl import flags

import gym as legacy_gym
from gfootball.env import config as env_config
from gfootball.env import football_env as raw_football_env
from gfootball.env import observation_preprocessing
from gfootball.env import wrappers as legacy_wrappers
import numpy as np

try:
  import gymnasium as gym
  import torch as th
  from torch import nn
  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import CheckpointCallback
  from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
  from stable_baselines3.common.vec_env import (
      DummyVecEnv,
      SubprocVecEnv,
      VecMonitor,
      VecTransposeImage,
  )
except ImportError as exc:
  raise ImportError(
      'Modern training example requires `stable-baselines3`, `gymnasium`, '
      'and `torch`. Install them with `uv pip install stable-baselines3 gymnasium '
      'shimmy`.'
  ) from exc


FLAGS = flags.FLAGS

flags.DEFINE_string('level', 'academy_empty_goal_close',
                    'Defines type of problem being solved.')
flags.DEFINE_enum('state', 'extracted_stacked',
                  ['extracted', 'extracted_stacked', 'simple115',
                   'simple115v2', 'pixels', 'pixels_gray'],
                  'Observation representation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'gfootball_impala_cnn',
                  ['cnn', 'mlp', 'impala_cnn', 'gfootball_impala_cnn'],
                  'Policy architecture. `impala_cnn` and '
                  '`gfootball_impala_cnn` use a residual CNN extractor.')
flags.DEFINE_integer('num_timesteps', int(2e6),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 128,
                     'Number of vectorized environment steps per rollout.')
flags.DEFINE_integer('noptepochs', 4, 'Number of optimization epochs.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one rollout into.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently to save checkpoints, in rollout updates.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.00008, 'Learning rate.')
flags.DEFINE_float('ent_coef', 0.01, 'Entropy coefficient.')
flags.DEFINE_float('gamma', 0.993, 'Discount factor.')
flags.DEFINE_float('cliprange', 0.27, 'PPO clip range.')
flags.DEFINE_float('max_grad_norm', 0.5, 'Max gradient norm.')
flags.DEFINE_bool('render', False, 'If True, enable rendering for the first env.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', None,
                    'Path to load an existing Stable-Baselines3 PPO model from.')
flags.DEFINE_string('logdir', 'ppo_logs',
                    'Directory to store TensorBoard logs, monitors, and models.')
flags.DEFINE_string('device', 'auto',
                    'Torch device to use (e.g. `auto`, `cpu`, `cuda`).')


def _convert_space(space):
  """Convert legacy gym spaces to gymnasium spaces."""
  if isinstance(space, legacy_gym.spaces.Box):
    return gym.spaces.Box(
        low=np.array(space.low, copy=True),
        high=np.array(space.high, copy=True),
        shape=space.shape,
        dtype=space.dtype)
  if isinstance(space, legacy_gym.spaces.Discrete):
    return gym.spaces.Discrete(space.n)
  if isinstance(space, legacy_gym.spaces.MultiDiscrete):
    return gym.spaces.MultiDiscrete(np.array(space.nvec, copy=True))
  if isinstance(space, legacy_gym.spaces.MultiBinary):
    return gym.spaces.MultiBinary(space.n)
  if isinstance(space, legacy_gym.spaces.Tuple):
    return gym.spaces.Tuple(tuple(_convert_space(s) for s in space.spaces))
  if isinstance(space, legacy_gym.spaces.Dict):
    return gym.spaces.Dict({
        key: _convert_space(value) for key, value in space.spaces.items()
    })
  raise TypeError('Unsupported legacy gym space: {}'.format(type(space)))


class _ResidualBlock(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self._conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                            padding=1)
    self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1,
                            padding=1)

  def forward(self, x):
    residual = x
    x = th.relu(x)
    x = self._conv1(x)
    x = th.relu(x)
    x = self._conv2(x)
    return x + residual


class GFootballImpalaCNN(BaseFeaturesExtractor):
  """PyTorch version of the paper's IMPALA-like CNN encoder."""

  def __init__(self, observation_space, features_dim=256):
    super().__init__(observation_space, features_dim)
    channels = observation_space.shape[0]
    conv_layers = ((16, 2), (32, 2), (32, 2), (32, 2))
    blocks = []
    in_channels = channels
    for out_channels, num_blocks in conv_layers:
      blocks.append(nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1))
      blocks.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
      for _ in range(num_blocks):
        blocks.append(_ResidualBlock(out_channels))
      in_channels = out_channels
    self._cnn = nn.Sequential(*blocks, nn.ReLU(), nn.Flatten())
    with th.no_grad():
      sample = th.as_tensor(
          observation_space.sample()[None]).float()
      n_flatten = self._cnn(sample).shape[1]
    self._linear = nn.Sequential(
        nn.Linear(n_flatten, features_dim),
        nn.ReLU(),
    )

  def forward(self, observations):
    return self._linear(self._cnn(observations))


class ModernFootballEnv(gym.Env):
  """Single-agent Gymnasium wrapper built directly on raw FootballEnv."""

  metadata = {'render_modes': ['human']}

  def __init__(self, rank, logdir, train_config):
    self._rank = rank
    self._train_config = train_config
    self._state = train_config['state']
    self._representation = train_config['representation']
    self._stacked = train_config['stacked']
    self._channel_dimensions = (
        observation_preprocessing.SMM_WIDTH,
        observation_preprocessing.SMM_HEIGHT)
    self._render_enabled = train_config['render'] or self._representation.startswith(
        'pixels')
    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1
    self._collected_checkpoints = 0
    self._frame_stack = collections.deque(
        [], maxlen=4 if self._stacked else 1)

    config_values = {
        'dump_full_episodes': train_config['dump_full_episodes'] and rank == 0,
        'dump_scores': train_config['dump_scores'] and rank == 0,
        'players': ['agent:left_players=1,right_players=0'],
        'level': train_config['level'],
        'tracesdir': str(logdir),
        'write_video': False,
    }
    self._config = env_config.Config(config_values)
    self._env = raw_football_env.FootballEnv(self._config)
    if self._render_enabled and rank == 0:
      self._env.render()

    self.action_space = _convert_space(self._env.action_space)
    self.observation_space = self._build_observation_space()

  def _build_observation_space(self):
    width, height = self._channel_dimensions
    if self._representation in ('pixels', 'pixels_gray'):
      channels = 1 if self._representation == 'pixels_gray' else 3
      if self._stacked:
        channels *= 4
      return gym.spaces.Box(
          low=0,
          high=255,
          shape=(height, width, channels),
          dtype=np.uint8)
    if self._representation == 'extracted':
      channels = 4 * (4 if self._stacked else 1)
      return gym.spaces.Box(
          low=0,
          high=255,
          shape=(height, width, channels),
          dtype=np.uint8)
    if self._representation == 'simple115':
      return gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
    if self._representation == 'simple115v2':
      return gym.spaces.Box(
          low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
    raise ValueError('Unsupported state representation: {}'.format(
        self._state))

  def _reset_frame_stack(self, observation):
    self._frame_stack.clear()
    self._frame_stack.extend([observation] * self._frame_stack.maxlen)
    return self._get_stacked_observation()

  def _get_stacked_observation(self):
    if not self._stacked:
      return self._frame_stack[-1]
    return np.concatenate(list(self._frame_stack), axis=-1)

  def _convert_observation(self, raw_observation):
    if self._representation == 'simple115':
      return legacy_wrappers.Simple115StateWrapper.convert_observation(
          raw_observation, False)[0]
    if self._representation == 'simple115v2':
      return legacy_wrappers.Simple115StateWrapper.convert_observation(
          raw_observation, True)[0]
    if self._representation == 'extracted':
      return observation_preprocessing.generate_smm(
          raw_observation,
          config=self._config,
          channel_dimensions=self._channel_dimensions)[0]
    if self._representation in ('pixels', 'pixels_gray'):
      frame = raw_observation[0]['frame']
      if self._representation == 'pixels_gray':
        import cv2
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      import cv2
      frame = cv2.resize(
          frame,
          self._channel_dimensions,
          interpolation=cv2.INTER_AREA)
      if self._representation == 'pixels_gray':
        frame = np.expand_dims(frame, -1)
      return np.array(frame, dtype=np.uint8)
    raise ValueError('Unsupported state representation: {}'.format(
        self._state))

  def _apply_checkpoint_reward(self, reward, raw_observation):
    if 'checkpoints' not in self._train_config['reward_experiment'].split(','):
      return reward
    if reward == 1:
      reward += self._checkpoint_reward * (
          self._num_checkpoints - self._collected_checkpoints)
      self._collected_checkpoints = self._num_checkpoints
      return reward
    observation = raw_observation[0]
    if ('ball_owned_team' not in observation or
        observation['ball_owned_team'] != 0 or
        'ball_owned_player' not in observation or
        observation['ball_owned_player'] != observation['active']):
      return reward
    distance = ((observation['ball'][0] - 1) ** 2 +
                observation['ball'][1] ** 2) ** 0.5
    while self._collected_checkpoints < self._num_checkpoints:
      if self._num_checkpoints == 1:
        threshold = 0.99 - 0.8
      else:
        threshold = (
            0.99 - 0.8 / (self._num_checkpoints - 1) *
            self._collected_checkpoints)
      if distance > threshold:
        break
      reward += self._checkpoint_reward
      self._collected_checkpoints += 1
    return reward

  def reset(self, *, seed=None, options=None):
    del options
    if seed is not None and hasattr(self._env, 'seed'):
      self._env.seed(seed)
    self._collected_checkpoints = 0
    raw_observation = self._env.reset()
    observation = self._convert_observation(raw_observation)
    if self._stacked:
      observation = self._reset_frame_stack(observation)
    else:
      self._frame_stack.clear()
      self._frame_stack.append(observation)
    return observation, {}

  def step(self, action):
    raw_observation, reward, done, info = self._env.step(action)
    reward = float(np.asarray(reward).reshape(-1)[0])
    reward = self._apply_checkpoint_reward(reward, raw_observation)
    observation = self._convert_observation(raw_observation)
    if self._stacked:
      self._frame_stack.append(observation)
      observation = self._get_stacked_observation()
    terminated = bool(done)
    truncated = bool(info.get('TimeLimit.truncated', False))
    return observation, reward, terminated and not truncated, truncated, info

  def render(self):
    return self._env.render()

  def close(self):
    return self._env.close()

  @property
  def unwrapped(self):
    return self._env.unwrapped


def _representation(state):
  if state.endswith('_stacked'):
    return state[:-len('_stacked')]
  return state


def _uses_stacking(state):
  return state.endswith('_stacked')


def _is_image_like(space):
  return isinstance(space, gym.spaces.Box) and len(space.shape) == 3


def _policy_name(policy):
  if policy == 'mlp':
    return 'MlpPolicy'
  return 'CnnPolicy'


def _policy_kwargs(policy):
  if policy in ('impala_cnn', 'gfootball_impala_cnn'):
    return {'features_extractor_class': GFootballImpalaCNN}
  return {}


def _make_env(rank, logdir, train_config):
  """Create one Gymnasium-compatible football env."""
  def _init():
    return ModernFootballEnv(rank, logdir, train_config)
  return _init


def _create_vec_env(logdir, train_config):
  env_fns = [_make_env(i, logdir, train_config)
             for i in range(train_config['num_envs'])]
  if train_config['num_envs'] == 1:
    vec_env = DummyVecEnv(env_fns)
  else:
    vec_env = SubprocVecEnv(env_fns)
  vec_env = VecMonitor(vec_env, filename=str(logdir / 'vec_monitor.csv'))
  if _is_image_like(vec_env.observation_space):
    vec_env = VecTransposeImage(vec_env)
  return vec_env


def train(_):
  """Train a PPO policy with Stable-Baselines3."""
  train_config = {
      'level': FLAGS.level,
      'state': FLAGS.state,
      'representation': _representation(FLAGS.state),
      'stacked': _uses_stacking(FLAGS.state),
      'reward_experiment': FLAGS.reward_experiment,
      'render': FLAGS.render,
      'dump_full_episodes': FLAGS.dump_full_episodes,
      'dump_scores': FLAGS.dump_scores,
      'num_envs': FLAGS.num_envs,
      'policy': FLAGS.policy,
  }
  rollout_size = FLAGS.num_envs * FLAGS.nsteps
  if rollout_size % FLAGS.nminibatches != 0:
    raise ValueError(
        'num_envs * nsteps ({}) must be divisible by nminibatches ({})'.format(
            rollout_size, FLAGS.nminibatches))

  logdir = Path(FLAGS.logdir).resolve()
  logdir.mkdir(parents=True, exist_ok=True)
  vec_env = _create_vec_env(logdir, train_config)
  batch_size = rollout_size // FLAGS.nminibatches
  checkpoint_dir = logdir / 'checkpoints'
  checkpoint_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_callback = CheckpointCallback(
      save_freq=max(FLAGS.save_interval * FLAGS.nsteps, 1),
      save_path=str(checkpoint_dir),
      name_prefix='ppo_model')

  if FLAGS.load_path:
    model = PPO.load(
        FLAGS.load_path,
        env=vec_env,
        device=FLAGS.device,
        print_system_info=True,
    )
  else:
    model = PPO(
        _policy_name(train_config['policy']),
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
        policy_kwargs=_policy_kwargs(train_config['policy']),
    )

  try:
    model.learn(
        total_timesteps=FLAGS.num_timesteps,
        callback=checkpoint_callback,
    )
    model.save(str(logdir / 'final_model'))
  finally:
    vec_env.close()


if __name__ == '__main__':
  app.run(train)
