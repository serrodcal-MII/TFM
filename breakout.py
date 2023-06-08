### IMPORTS
import time
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

tf.get_logger().setLevel("INFO")

### ENV DEFINITION
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.wrappers.AtariPreprocessing(
    env, screen_size=84, frame_skip=1, grayscale_obs=True
)
env = gym.wrappers.FrameStack(env, 4)

num_actions = env.action_space.n


### DQN
def build_dqn_model():
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
    dense = layers.Dense(512, activation="relu")(flat)
    action = layers.Dense(num_actions, activation="linear")(dense)

    return keras.Model(inputs=inputs, outputs=action)


model = build_dqn_model()
model_target = build_dqn_model()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

loss_function = keras.losses.Huber()

### PARAMETERS
discount_factor = 0.99

epsilon = 1.0

epsilon_range_1 = (1.0, 0.2)
epsilon_range_2 = (0.2, 0.1)
epsilon_range_3 = (0.1, 0.02)

epsilon_greedy_frames = 1000000.0

random_frames_threshold = 50000

max_buffer_length = 200000

batch_size = 32
steps_per_episode = 10000

update_after_actions = 20

update_target_network = 10000


### UTILS
def transpose_state(state):
    new_state = np.transpose(
        np.array(state), (1, 2, 0)
    )  # Convert from LazyFrame to numpy.ndarray and Transpose from (4, 84, 84) to (84, 84, 4)
    return new_state


def update_epsilon_greedy(e, frame):
    if frame < epsilon_greedy_frames:
        e -= (epsilon_range_1[0] - epsilon_range_1[1]) / epsilon_greedy_frames
        return max(e, 0.2)
    elif frame > epsilon_greedy_frames and frame < 2 * epsilon_greedy_frames:
        e -= (epsilon_range_2[0] - epsilon_range_2[1]) / epsilon_greedy_frames
        return max(e, 0.1)
    else:
        e -= (epsilon_range_3[0] - epsilon_range_3[1]) / epsilon_greedy_frames
        return max(e, 0.02)


### EXPERIMENT
action_buffer = []
state_buffer = []
state_next_buffer = []
rewards_buffer = []
terminated_buffer = []

episodes_rewards = []
running_reward = 0
episodes = 0
frames = 0

while True:
    state, _ = env.reset()
    state = transpose_state(state)
    episode_reward = 0

    for _ in range(1, steps_per_episode):
        frames += 1

        if frames < random_frames_threshold or epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        epsilon = update_epsilon_greedy(epsilon, frames)

        state_next, reward, terminated, _, _ = env.step(action)
        state_next = transpose_state(state_next)

        action_buffer.append(action)
        state_buffer.append(state)
        state_next_buffer.append(state_next)
        terminated_buffer.append(terminated)
        rewards_buffer.append(reward)
        state = state_next

        episode_reward += reward

        if frames % update_after_actions == 0 and len(terminated_buffer) > batch_size:
            indices = np.random.choice(range(len(terminated_buffer)), size=batch_size)

            state_sample = np.array([state_buffer[i] for i in indices])
            state_next_sample = np.array([state_next_buffer[i] for i in indices])
            rewards_sample = [rewards_buffer[i] for i in indices]
            action_sample = [action_buffer[i] for i in indices]
            terminated_buffersample = tf.convert_to_tensor(
                [float(terminated_buffer[i]) for i in indices]
            )

            next_state_rewards = model_target.predict(state_next_sample, verbose=0)

            updated_q_values = (
                rewards_sample
                + discount_factor * tf.reduce_max(next_state_rewards, axis=1)
            ) * ((1 - terminated_buffersample) - terminated_buffersample)

            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                q_values = model(state_sample)
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = loss_function(updated_q_values, q_action)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frames % update_target_network == 0:
            model_target.set_weights(model.get_weights())
            print(
                f"running reward: {running_reward:.2f} at episode {episodes}, frame count {frames}, epsilon {epsilon:.3f}, loss {loss:.5f}"
            )

        if len(rewards_buffer) > max_buffer_length:
            del rewards_buffer[:1]
            del state_buffer[:1]
            del state_next_buffer[:1]
            del action_buffer[:1]
            del terminated_buffer[:1]

        if terminated:
            break

    episodes_rewards.append(episode_reward)
    if len(episodes_rewards) > 100:
        del episodes_rewards[:1]
    running_reward = np.mean(episodes_rewards)

    episodes += 1

    if running_reward > 18:
        print(f"Solved at episode {episodes}!")
        break


current_ts = time.strftime("%m-%d-%Y_%H:%M:%S", time.gmtime())

model.save("breakout_" + current_ts)
model_target.save("target_breakout_" + current_ts)
