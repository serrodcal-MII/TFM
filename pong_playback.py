import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from datetime import datetime

model_path = "pong_model"

current_time = datetime.now()
timestamp = str(current_time).split(".")[0]

video_path = "video_pong_" + "".join(c for c in timestamp if c.isalnum())

env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
env = gym.wrappers.GrayScaleObservation(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))
env = gym.wrappers.FrameStack(env, 4)
env = gym.wrappers.RecordVideo(env, video_path)

num_actions = env.action_space.n

model = tf.keras.models.load_model(model_path)

_, _ = env.reset()
observation, _, _, _, _ = env.step(1)
observation = np.array(observation)  # Convert from LazyFrame to numpy.ndarray
observation = np.transpose(observation, (1, 2, 0))
info = 0
reward_window = []
reward_signal_history = []
epsilon_history = []

hits = []
bltd = 108  # total bricks to destroy

for i_episode in range(1):
    reward_window = []
    epsilon = 0
    for t in range(4000):
        if epsilon > np.random.rand(1)[0]:
            action = np.random.choice(num_actions)
        else:
            state_tensor = tf.convert_to_tensor(observation)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

        observation, reward, terminated, _, _ = env.step(action)
        observation = np.array(observation)  # Convert from LazyFrame to numpy.ndarray
        observation = np.transpose(observation, (1, 2, 0))
        hits.append(reward)
        reward_window.append(reward)
        if len(reward_window) > 200:
            del reward_window[:1]
        if len(reward_window) == 200 and np.sum(reward_window) == 0:
            epsilon = 0.01
        else:
            epsilon = 0.0001

        epsilon_history.append(epsilon)
        reward_signal_history.append(reward)

        if terminated:
            print("Lost one life after {} timesteps".format(t + 1))
            print(info)
            # Plot epsilon and reward signal
            fig, ax = plt.subplots(figsize=(20, 3))
            # plt.clf()
            ax.plot(epsilon_history, color="red")
            ax.set_ylabel("epsilon", color="red", fontsize=14)
            ax2 = ax.twinx()
            ax2.plot(reward_signal_history, color="blue")
            ax2.set_ylabel("reward_signal", color="blue", fontsize=14)
            plt.show()

            epsilon_history = []
            reward_signal_history = []

            bltd = bltd - np.sum(hits)
            hits = []
            print("Bricks left to destroy ", bltd)
            if info == 0:
                break

            env.reset()

env.close()
