import math

import numpy as np
import pyglet
from pyglet.window import key


class ManualControl:
    def __init__(self, env, no_time_limit: bool, domain_rand: bool, heading_zero: bool = False):
        self.env = env.unwrapped
        self.heading_zero = heading_zero
        self.active_agent_idx = 0

        if no_time_limit:
            self.env.max_episode_steps = math.inf
        if domain_rand:
            self.env.domain_rand = True

    def run(self):
        print("============")
        print("Instructions")
        print("============")
        print(
            "select agent: 1-4\nmove: arrow keys\npickup: P\n"
            "drop: D\ndone: ENTER\nquit: ESC"
        )
        print("============")

        self.env.reset()
        if self.heading_zero:
            self._align_headings()

        # Create the display window
        self.env.render()

        env = self.env

        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            """
            This handler processes keyboard commands that
            control the simulation
            """

            if symbol == key.BACKSPACE or symbol == key.SLASH:
                print("RESET")
                self.env.reset()
                if self.heading_zero:
                    self._align_headings()
                self.env.render()
                return

            if symbol == key.ESCAPE:
                self.env.close()

            number_keys = (key._1, key._2, key._3, key._4)
            if symbol in number_keys:
                idx = number_keys.index(symbol)
                if idx < len(self.env.agents):
                    self.active_agent_idx = idx
                    print(f"selected agent {idx + 1}")
                return

            if symbol == key.UP:
                self.step(self.env.unwrapped.actions.move_forward)
            elif symbol == key.DOWN:
                self.step(self.env.unwrapped.actions.move_back)
            elif symbol == key.LEFT:
                self.step(self.env.unwrapped.actions.turn_left)
            elif symbol == key.RIGHT:
                self.step(self.env.unwrapped.actions.turn_right)
            elif symbol == key.PAGEUP or symbol == key.P:
                self.step(self.env.unwrapped.actions.pickup)
            elif symbol == key.PAGEDOWN or symbol == key.D:
                self.step(self.env.unwrapped.actions.drop)
            elif symbol == key.ENTER:
                self.step(self.env.unwrapped.actions.done)

        @env.unwrapped.window.event
        def on_key_release(symbol, modifiers):
            pass

        @env.unwrapped.window.event
        def on_draw():
            self.env.render()

        @env.unwrapped.window.event
        def on_close():
            pyglet.app.exit()

        # Enter main event loop
        pyglet.app.run()

        self.env.close()

    def step(self, action):
        agent_label = self.active_agent_idx + 1
        print(
            f"step {self.env.unwrapped.step_count + 1}/{self.env.unwrapped.max_episode_steps}: "
            f"agent {agent_label} {self.env.unwrapped.actions(action).name}"
        )

        if self.env.num_agents == 1:
            joint_action = action
        else:
            joint_action = np.full(
                self.env.num_agents,
                self.env.actions.do_nothing,
                dtype=np.int64,
            )
            joint_action[self.active_agent_idx] = action
        obs, reward, termination, truncation, info = self.env.step(joint_action)

        if reward > 0:
            print(f"reward={reward:.2f}")

        if termination or truncation:
            print("done!")
            self.env.reset()
            if self.heading_zero:
                self._align_headings()

        self.env.render()

    def _align_headings(self):
        for agent in self.env.agents:
            agent.dir = 0.0
            if agent.carrying:
                agent.carrying.dir = agent.dir
