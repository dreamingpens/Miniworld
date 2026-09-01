import math
import pickle
import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env, data_equivalence

import miniworld
from miniworld.entity import TextFrame
from miniworld.miniworld import MiniWorldEnv
from miniworld.wrappers import PyTorchObsWrapper, StochasticActionWrapper

miniworld_env_ids = [env_id for env_id in gym.registry if "MiniWorld" in env_id]


def test_miniworld():
    env = gym.make("MiniWorld-Hallway-v0", render_mode="rgb_array")

    env.reset()
    # Try stepping a few times
    for i in range(0, 10):
        obs, _, _, _, _ = env.step(0)

    # Check that the human rendering resembles the agent's view
    first_obs, info = env.reset()
    first_render = env.render()
    m0 = first_obs.mean()
    m1 = first_render.mean()
    assert 0 < m0 < 255
    assert abs(m0 - m1) < 5

    # Check that the observation shapes match in reset and step
    second_obs, _, _, _, _ = env.step(0)
    assert first_obs.shape == env.observation_space.shape
    assert first_obs.shape == second_obs.shape

    env.close()


def test_pytorch_wrapper():
    env = gym.make("MiniWorld-Hallway-v0")
    # Test the PyTorch observation wrapper
    env = PyTorchObsWrapper(env)
    first_obs, info = env.reset()
    second_obs, _, _, _, _ = env.step(0)
    assert first_obs.shape == env.observation_space.shape
    assert first_obs.shape == second_obs.shape

    env.close()


def test_stochastic_wrapper():
    env = gym.make("MiniWorld-Hallway-v0")
    # Test the stochastic action wrapper
    env = StochasticActionWrapper(env, prob=0.9)
    _, _ = env.reset()
    _, _, _, _, _ = env.step(0)
    env.close()

    env = StochasticActionWrapper(env, prob=0.9, random_action=1)
    _, _ = env.reset()
    _, _, _, _, _ = env.step(0)
    env.close()


def test_text_frame():
    # Test TextFrame
    # make sure it loads the TextFrame with no issues
    class TestText(miniworld.envs.ThreeRooms):
        def _gen_world(self):
            super()._gen_world()
            self.entities.append(
                TextFrame(pos=[0, 1.35, 7], dir=math.pi / 2, str="this is a test")
            )

    env = TestText()
    env.reset()
    env.close()


def test_collision_detection():
    # Basic collision detection test
    # Make sure the agent can never get outside the room
    env = gym.make("MiniWorld-OneRoom-v0").env.env
    for _ in range(30):
        env.reset()
        room = env.rooms[0]
        for _ in range(30):
            env.step(env.actions.move_forward)
            x, _, z = env.agent.pos
            assert room.min_x <= x <= room.max_x
            assert room.min_z <= z <= room.max_z

    env.close()


def test_static_scenery_and_fixed_camera():
    env = gym.make(
        "MiniWorld-MovingBlockWorld-v0",
        render_mode="rgb_array",
        obs_width=32,
        obs_height=32,
        window_width=32,
        window_height=32,
        size=16,
        num_blocks=8,
        num_static_objects=12,
        static_object_spacing=3.0,
    )
    env.reset(seed=7)
    world = env.unwrapped

    assert len(world.static_scenery) == 12
    assert world.static_scenery[0].pos.tolist() == [8.0, 0.0, 8.0]

    frame = world.render_camera(
        eye_pos=[1.0, 8.0, 1.0],
        target_pos=[8.0, 0.75, 8.0],
        fov_y=60.0,
    )
    assert frame.shape == (32, 32, 3)
    assert 0 < frame.mean() < 255

    env.close()


@pytest.mark.parametrize("num_agents", [3, 4])
def test_multi_agent_observations_and_actions(num_agents):
    env = gym.make(
        "MiniWorld-MovingBlockWorld-v0",
        render_mode="rgb_array",
        obs_width=32,
        obs_height=24,
        window_width=32,
        window_height=24,
        size=16,
        num_blocks=2,
        blocks_static=True,
        num_agents=num_agents,
    )
    obs, _ = env.reset(seed=11)
    world = env.unwrapped

    assert len(world.agents) == num_agents
    assert obs.shape == (num_agents, 24, 32, 3)
    assert env.action_space.shape == (num_agents,)
    assert len({tuple(agent.pos) for agent in world.agents}) == num_agents

    actions = np.full(num_agents, world.actions.do_nothing, dtype=np.int64)
    actions[0] = world.actions.turn_left
    actions[1] = world.actions.turn_right
    next_obs, _, _, _, _ = env.step(actions)
    assert next_obs.shape == obs.shape

    views = world.render_agent_views(world.vis_fb)
    assert views.shape == (num_agents, 24, 32, 3)
    montage = world.tile_agent_views(views)
    assert montage.shape == (48, 64, 3)
    assert 0 < montage.mean() < 255

    env.close()


@pytest.mark.parametrize("env_id", miniworld_env_ids)
def test_all_envs(env_id):
    # Try loading each of the available environments
    if "RemoteBot" in env_id:
        return

    env = gym.make(env_id)
    assert isinstance(env.unwrapped, MiniWorldEnv)
    env = env.env.env

    env.domain_rand = True
    # Try multiple random restarts
    for _ in range(15):
        env.reset()
        assert not env.intersect(env.agent, env.agent.pos, env.agent.radius)
        # Perform multiple random actions
        for _ in range(0, 20):
            # Change to the attribute of np_rand
            action = env.np_random.integers(0, env.action_space.n)
            obs, reward, done, truncation, info = env.step(action)
            if done:
                env.reset()
    env.close()


CHECK_ENV_IGNORE_WARNINGS = [
    f"\x1b[33mWARN: {message}\x1b[0m"
    for message in [
        "A Box observation space minimum value is -infinity. This is probably too low.",
        "A Box observation space maximum value is -infinity. This is probably too high.",
        "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        "Initializing wrapper in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.",
        "Initializing environment in old step API which returns one bool instead of two. It is recommended to set `new_step_api=True` to use new step API. This will be the default behaviour in future.",
        "Core environment is written in old step API which returns one bool instead of two. It is recommended to rewrite the environment with new step API. ",
    ]
]


@pytest.mark.parametrize("env_id", miniworld_env_ids)
def test_env_checker(env_id):
    if "RemoteBot" in env_id:
        return

    env = gym.make(env_id).unwrapped
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
        check_env(env, skip_render_check=True)

    for warning in w:
        if warning.message.args[0] not in CHECK_ENV_IGNORE_WARNINGS:
            raise gym.error.Error(f"Unexpected warning: {warning.message}")

    env.close()


@pytest.mark.parametrize("env_id", miniworld_env_ids)
def test_pickle_env(env_id):
    if "RemoteBot" in env_id:
        return

    env = gym.make(env_id, max_episode_steps=100).unwrapped
    pickled_env = pickle.loads(pickle.dumps(env))

    data_equivalence(env.reset(), pickled_env.reset())
    action = env.action_space.sample()
    data_equivalence(env.step(action), pickled_env.step(action))

    env.close()
    pickled_env.close()
