### IMPORTS
import time
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from typing import List, Tuple, Any

tf.get_logger().setLevel("INFO")

### GLOBAL PARAMETERS
discount_factor = 0.97

epsilon_range_1 = (1.0, 0.2)
epsilon_range_2 = (0.2, 0.1)
epsilon_range_3 = (0.1, 0.02)

epsilon_greedy_iters = 1000000.0
random_iters_threshold = 50000
max_buffer_length = 200000
batch_size = 32
steps_per_episode = 10000
train_after_actions = 20
update_target_model_frequency = 10000
solve_condition = -7


### UTILS
def build_env(
    game: str = "ALE/Pong-v5",
    render_mode: str = "rgb_array",
    screen_size: int = 84,
    frame_skip: int = 1,
    grayscale_obs: bool = True,
) -> Any:
    env = gym.make(game, render_mode=render_mode)
    env = gym.wrappers.AtariPreprocessing(
        env, screen_size=screen_size, frame_skip=frame_skip, grayscale_obs=grayscale_obs
    )
    env = gym.wrappers.FrameStack(env, 4)

    return env


def preprocess(state: np.ndarray) -> np.ndarray:
    # Convert from LazyFrame to numpy.ndarray and Transpose from (4, 84, 84) to (84, 84, 4)
    transposed = np.transpose(np.array(state), (1, 2, 0))
    return transposed


def update_epsilon_greedy(e: float, frame: int) -> float:
    if frame < epsilon_greedy_iters:
        e -= (epsilon_range_1[0] - epsilon_range_1[1]) / epsilon_greedy_iters
        return max(e, 0.2)
    elif frame > epsilon_greedy_iters and frame < 2 * epsilon_greedy_iters:
        e -= (epsilon_range_2[0] - epsilon_range_2[1]) / epsilon_greedy_iters
        return max(e, 0.1)
    else:
        e -= (epsilon_range_3[0] - epsilon_range_3[1]) / epsilon_greedy_iters
        return max(e, 0.02)


def unzip_replay_bufer(buffer: List[Tuple], sampling_indices: np.array) -> Tuple:
    state_sample = np.array([buffer[i][1] for i in sampling_indices])
    next_state_sample = np.array([buffer[i][2] for i in sampling_indices])
    rewards_sample = [buffer[i][4] for i in sampling_indices]
    action_sample = [buffer[i][0] for i in sampling_indices]
    terminated_sample = tf.convert_to_tensor(
        [float(buffer[i][3]) for i in sampling_indices]
    )

    return (
        state_sample,
        next_state_sample,
        rewards_sample,
        action_sample,
        terminated_sample,
    )


### Agent
class Agent:
    def __init__(self, n_actions: int, lr: float = 0.00025) -> Any:
        self.n_actions = n_actions
        self.model = self.build_dqn_model(n_actions)
        self.target_model = self.build_dqn_model(n_actions)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()

    ### DQN
    def build_dqn_model(self, n_actions: int) -> Any:
        inputs = layers.Input(
            shape=(
                84,
                84,
                4,
            )
        )

        conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
        conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")(conv1)
        conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")(conv2)
        flat = layers.Flatten()(conv3)
        dense = layers.Dense(128, activation="relu")(flat)
        output = layers.Dense(n_actions, activation="linear")(dense)

        return keras.Model(inputs=inputs, outputs=output)

    def train(self, buffer: List[Tuple]) -> Any:
        sampling_indices = np.random.choice(range(len(buffer)), size=batch_size)

        (
            state_sample,
            next_state_sample,
            rewards_sample,
            action_sample,
            terminated_sample,
        ) = unzip_replay_bufer(buffer=buffer, sampling_indices=sampling_indices)

        next_state_rewards = self.target_model.predict(next_state_sample, verbose=0)

        next_q_value = tf.reduce_max(next_state_rewards, axis=1)

        updated_q_values = (rewards_sample + discount_factor * next_q_value) * (
            1 - terminated_sample
        )

        masks = tf.one_hot(action_sample, num_actions)

        with tf.GradientTape() as tape:
            q_values = self.model(state_sample)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = self.loss_function(updated_q_values, q_action)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

    def next_action(self, state: np.ndarray, iters: int, epsilon: float) -> int | float:
        actuation_protocol = (
            "exploration"
            if iters < random_iters_threshold or epsilon > np.random.rand(1)[0]
            else "explotation"
        )

        if actuation_protocol == "exploration":
            return np.random.choice(num_actions)
        else:
            return self.exploit(state)

    def exploit(self, state: np.ndarray) -> int:
        state_tensor = tf.expand_dims(tf.convert_to_tensor(state), 0)
        action_probs = self.model(state_tensor, training=False)
        best_action = tf.argmax(action_probs[0]).numpy()

        return best_action

    def update_target_model(self) -> None:
        self.target_model.set_weights(self.model.get_weights())


### EXPERIMENT
def run(env: Any, agent: Agent) -> None:
    replay_buffer = []

    episodes_rewards = []
    running_reward = -20
    episodes = 0
    iters = 0

    epsilon = 1.0

    while running_reward < solve_condition:
        state, _ = env.reset()
        state = preprocess(state)
        episode_reward = 0

        for _ in range(1, steps_per_episode):
            iters += 1

            action = agent.next_action(state, iters, epsilon)

            epsilon = update_epsilon_greedy(epsilon, iters)

            next_state, reward, terminated, _, _ = env.step(action)
            next_state = preprocess(next_state)

            replay_buffer.append((action, state, next_state, terminated, reward))

            state = next_state
            episode_reward += reward

            if iters % train_after_actions == 0 and len(replay_buffer) > batch_size:
                loss = agent.train(replay_buffer)

            if iters % update_target_model_frequency == 0:
                agent.update_target_model()
                print(
                    f"running reward: {running_reward:.2f} at episode {episodes}, frame count {iters}, epsilon {epsilon:.3f}, loss {loss:.5f}"
                )

            if len(replay_buffer) > max_buffer_length:
                del replay_buffer[:1]

            if terminated:
                break

        episodes_rewards.append(episode_reward)
        if len(episodes_rewards) > 100:
            del episodes_rewards[:1]
        running_reward = np.mean(episodes_rewards)

        episodes += 1

    print(f"Solved at episode {episodes}!")


if __name__ == "__main__":
    ### ENV DEFINITION
    env = build_env(
        game="ALE/Pong-v5",
        render_mode="rgb_array",
        screen_size=84,
        frame_skip=1,
        grayscale_obs=True,
    )

    num_actions = env.action_space.n

    agent = Agent(num_actions)

    run(env, agent)

    current_ts = time.strftime("%m-%d-%Y_%H:%M:%S", time.gmtime())

    agent.model.save("pong_" + current_ts)
    agent.target_model.save("target_pong_" + current_ts)
