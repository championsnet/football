# Google Research Football

This repository contains an RL environment based on open-source game Gameplay
Football. <br> It was created by the Google Brain team for research purposes.

Useful links:

* [Run in Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb) - start training in less that 2 minutes.
* [Google Research Football Paper](https://arxiv.org/abs/1907.11180)
* [GoogleAI blog post](https://ai.googleblog.com/2019/06/introducing-google-research-football.html)
* [Google Research Football on Cloud](https://towardsdatascience.com/reproducing-google-research-football-rl-results-ac75cf17190e)
* [GRF Kaggle competition](https://www.kaggle.com/c/google-football) - take part in the competition playing games against others, win prizes and become the GRF Champion!


We'd like to thank Bastiaan Konings Schuiling, who authored and open-sourced the original version of this game.


## Quick Start

### In colab

Open our example [Colab](https://colab.research.google.com/github/google-research/football/blob/master/gfootball/colabs/gfootball_example_from_prebuild.ipynb), that will allow you to start training your model in less than 2 minutes.

This method doesn't support game rendering on screen - if you want to see the game running, please use the method below.

### Using Docker

This is the recommended way for Linux-based systems to avoid incompatible package versions.
Instructions are available [here](gfootball/doc/docker.md).

### On your computer

#### 1. Install required packages
#### Linux
```shell
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

python3 -m pip install --upgrade pip setuptools psutil wheel
```

#### macOS
First install [brew](https://brew.sh/). It should automatically install Command Line Tools.
Next install required packages:

```shell
brew install git python3 cmake sdl2 sdl2_image sdl2_ttf sdl2_gfx boost boost-python3

python3 -m pip install --upgrade pip setuptools psutil wheel
```


#### Windows
Install [Git](https://git-scm.com/download/win) and [Python 3](https://www.python.org/downloads/).
Update pip in the Command Line (here and for the **next steps** type `python` instead of `python3`)
```commandline
python -m pip install --upgrade pip setuptools psutil wheel
```


#### 2. Install GFootball
#### Option a. From PyPi package (recommended)
```shell
python3 -m pip install gfootball
```

#### Option b. Installing from sources using GitHub repository 
(On Windows you have to install additional tools and set an environment variable, see 
[Compiling Engine](gfootball/doc/compile_engine.md#windows) for detailed instructions.)

```shell
git clone https://github.com/google-research/football.git
cd football
```

Optionally you can use [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```shell
python3 -m venv football-env
source football-env/bin/activate
```

Or, with [uv](https://docs.astral.sh/uv/), create and manage the environment with:

```shell
uv venv --python 3.14 football-env
source football-env/bin/activate
```

The project currently supports Python 3.6+. On newer systems, pass `--python 3.14` to `uv` so it does not pick a mismatched interpreter automatically.

Next, build the game engine and install dependencies:

```shell
python3 -m pip install .
```
This command can run for a couple of minutes, as it compiles the C++ environment in the background.
If you face any problems, first check [Compiling Engine](gfootball/doc/compile_engine.md) documentation and search GitHub issues.

To install from source with `uv`, run:

```shell
uv sync --python 3.14
```

`uv sync` performs an editable install of the local project, so code changes are picked up without reinstalling. If you prefer the pip-style workflow, `uv pip install .` and `uv pip install -e .` are also supported.


#### 3. Time to play!
```shell
python3 -m gfootball.play_game --action_set=full
```
Make sure to check out the [keyboard mappings](#keyboard-mappings).
To quit the game press Ctrl+C in the terminal.

If you are using `uv` without activating the virtual environment, you can run:

```shell
uv run python -m gfootball.play_game --action_set=full
```

# Contents #

* [Running training](#training-agents-to-play-GRF)
* [Playing the game](#playing-the-game)
    * [Keyboard mappings](#keyboard-mappings)
    * [Play vs built-in AI](#play-vs-built-in-AI)
    * [Play vs pre-trained agent](#play-vs-pre-trained-agent)
    * [Trained checkpoints](#trained-checkpoints)
* [Environment API](gfootball/doc/api.md)
* [Observations & Actions](gfootball/doc/observation.md)
* [Scenarios](gfootball/doc/scenarios.md)
* [Multi-agent support](gfootball/doc/multi_agent.md)
* [Running in docker](gfootball/doc/docker.md)
* [Saving replays, logs, traces](gfootball/doc/saving_replays.md)
* [Imitation Learning](gfootball/doc/imitation.md)

## Training agents to play GRF

### Run training
The public training example now uses a modern PPO stack based on
Stable-Baselines3 instead of the legacy TensorFlow 1.15/OpenAI Baselines setup.

Install the extra training dependencies with:

- `uv pip install stable-baselines3 gymnasium shimmy`

Then:

- To run an example PPO experiment on `academy_empty_goal_close`, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_empty_goal_close`
- To run on `academy_pass_and_shoot_with_keeper`, run
  `python3 -m gfootball.examples.run_ppo2 --level=academy_pass_and_shoot_with_keeper`

The default policy is a PyTorch port of the repo's IMPALA-style CNN encoder.
You can also use a plain SB3 CNN or MLP with `--policy=cnn` or `--policy=mlp`.

In order to train with replays being saved, run
`python3 -m gfootball.examples.run_ppo2 --dump_full_episodes=True --render=True`

In order to reproduce the original legacy PPO2 results from the paper, please refer to:

- gfootball/examples/repro_checkpoint_easy.sh
- gfootball/examples/repro_scoring_easy.sh

### Run full 11 vs 11 training

To train a cooperative 11v11 agent system that simulates a full football match, use the dedicated script:

- `python3 -m gfootball.examples.run_11v11`

This trains a single **shared MLP policy** (parameter sharing) that is applied independently to each of the 11 left-team players. All players receive the same team reward (+1 goal scored, −1 goal conceded) making the setting fully cooperative. The right team is controlled by the built-in AI.

Key flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--level` | `11_vs_11_easy_stochastic` | Scenario (easy/hard/stochastic/competition) |
| `--num_games` | `4` | Parallel games; total VecEnv width = `num_games × 11` |
| `--num_timesteps` | `50_000_000` | Total environment steps |
| `--reward_experiment` | `scoring,checkpoints` | Add dense checkpoint shaping |
| `--load_path` | _(none)_ | Resume from an existing checkpoint |
| `--logdir` | `ppo_logs_11v11` | Output directory |

Example — harder opponent, more parallelism:

```bash
python3 -m gfootball.examples.run_11v11 \
  --level=11_vs_11_hard_stochastic \
  --num_games=8 \
  --num_timesteps=200000000
```

## Playing the game

Please note that playing the game is implemented through an environment, so human-controlled players use the same interface as the agents.
One important implication is that there is a single action per 100 ms reported to the environment, which might cause a lag effect when playing.


### Keyboard mappings
The game defines following keyboard mapping (for the `keyboard` player type):

* `ARROW UP` - run to the top.
* `ARROW DOWN` - run to the bottom.
* `ARROW LEFT` - run to the left.
* `ARROW RIGHT` - run to the right.
* `S` - short pass in the attack mode, pressure in the defense mode.
* `A` - high pass in the attack mode, sliding in the defense mode.
* `D` - shot in the attack mode, team pressure in the defense mode.
* `W` - long pass in the attack mode, goalkeeper pressure in the defense mode.
* `Q` - switch the active player in the defense mode.
* `C` - dribble in the attack mode.
* `E` - sprint.

### Play vs built-in AI
Run `python3 -m gfootball.play_game --action_set=full`. By default, it starts
the base scenario and the left player is controlled by the keyboard. Different
types of players are supported (gamepad, external bots, agents...). For possible
options run `python3 -m gfootball.play_game -helpfull`.

### Play vs pre-trained agent

In particular, one can play against agent trained with `run_ppo2` script with
the following command (notice no action_set flag, as PPO agent uses default
action set):
`python3 -m gfootball.play_game --players "keyboard:left_players=1;ppo2_cnn:right_players=1,checkpoint=$YOUR_PATH"`

### Trained checkpoints
We provide trained PPO checkpoints for the following scenarios:

  - [11_vs_11_easy_stochastic](https://storage.googleapis.com/gfootball-public-bucket/trained_model_11_vs_11_easy_stochastic),
  - [academy_run_to_score_with_keeper](https://storage.googleapis.com/gfootball-public-bucket/trained_model_academy_run_to_score_with_keeper_v2).

In order to see the checkpoints playing, run
`python3 -m gfootball.play_game --players "ppo2_cnn:left_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT" --level=$LEVEL`,
where `$CHECKPOINT` is the path to downloaded checkpoint. Please note that the checkpoints were trained with Tensorflow 1.15 version. Using 
different Tensorflow version may result in errors. The easiest way to run these checkpoints is through provided `Dockerfile_examples` image.
See [running in docker](gfootball/doc/docker.md) for details (just override the default Docker definition with `-f Dockerfile_examples` parameter).

In order to train against a checkpoint, you can pass 'extra_players' argument to create_environment function.
For example extra_players='ppo2_cnn:right_players=1,policy=gfootball_impala_cnn,checkpoint=$CHECKPOINT'.
