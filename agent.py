from collections import namedtuple
from datetime import datetime
import plotting
import gym
import numpy as np
import random
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt


def action_selection(state, q_network, episode, n_episodes, action_size):

    epsilon = max(0, episode/n_episodes*2)
    if np.random.random() < epsilon:
        action = np.random.randint(action_size)
        return action, epsilon
    else:
        # action = np.argmax(Q[state])

        action_values = q_network.predict(state)

    return np.argmax(action_values[0]), epsilon


def action_selection_decay(state, q_network, episode, n_episodes, action_size, decay=0.006, min_epsilon=0.0, initial=1.00):
    # source: Miguel Morales' notebook

    epsilon = initial * math.exp(-decay*episode)
    epsilon = max(epsilon, min_epsilon)

    if np.random.random() < epsilon:
        action = np.random.randint(action_size)
        return action, epsilon
    else:
        action_values = q_network.predict(state)

    return np.argmax(action_values[0]), epsilon


def get_targets(q_network, target_network, batch, gamma):

    for state, action, reward, next_state, done in batch:

        targets = target_network.predict(state)

        if done:
            targets[0][action] = reward

        else:
            next_state = np.reshape(next_state, [1, np.size(state)])
            targets[0][action] = reward + gamma * np.max(target_network.predict(next_state))

        q_network.fit(state, targets, verbose=0)

    return targets


def deep_q_learning(env, n_episodes=1000, alpha=0.5, gamma=0.9, initial_replay_memory_size=500, replay_memory_size=1000,
                    batch_size=10, decay=0.006, min_epsilon=0.0, tau=0.01):

    # print start time
    print('Start time:', str(datetime.now()))

    # prep work
    transition = namedtuple('transition', ['state', 'action', 'reward', 'next_state', 'done'])
    action_size = env.action_space.n

    state_size = env.observation_space.shape[0]

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(n_episodes), episode_rewards=np.zeros(n_episodes))

    # initialize Q-network
    q_network = Sequential()
    q_network.add(Dense(units=24, activation='relu', input_dim=state_size))
    q_network.add(Dense(units=24, activation='relu'))
    q_network.add(Dense(action_size, activation='linear'))
    q_network.compile(loss='mse', optimizer=Adam(lr=alpha))

    # initialize target-network
    target_network = Sequential()
    target_network.add(Dense(units=24, activation='relu', input_dim=state_size))
    target_network.add(Dense(units=24, activation='relu'))
    target_network.add(Dense(action_size, activation='linear'))
    target_network.compile(loss='mse', optimizer=Adam(lr=alpha))

    # initialize replay memory
    replay_memory = []

    # populate replay memory
    for episode in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            if len(replay_memory) == initial_replay_memory_size:
                break
            else:
                # select a random action
                state = np.reshape(state, [1, state_size])
                # action, eps = action_selection(state, q_network, episode, n_episodes, action_size)
                action, eps = action_selection_decay(state, q_network, episode, n_episodes, action_size, decay=decay,
                                                     min_epsilon=min_epsilon)

                # execute an action
                next_state, reward, done, info = env.step(action)

                # store transition to replay memory
                replay_memory.append(transition(state, action, reward, next_state, done))

                state = next_state

        if len(replay_memory) == initial_replay_memory_size:
            break

    # q-learning
    for episode in range(n_episodes):
        print('Episode:', episode)
        state = env.reset()
        done = False
        t_count = 0

        while not done:

            state = np.reshape(state, [1, state_size])

            # select an action
            # action, eps = action_selection(state, q_network, episode, n_episodes, action_size)
            action, eps = action_selection_decay(state, q_network, episode, n_episodes, action_size, decay=decay,
                                                     min_epsilon=min_epsilon)

            # execute an action
            next_state, reward, done, info = env.step(action)

            # if replay memory is full, pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)

            # store transition to replay memory
            replay_memory.append(transition(state, action, reward, next_state, done))

            # update statistics
            stats.episode_rewards[episode] += reward
            stats.episode_lengths[episode] = t_count

            # sample a random minibatch from replay memory
            mini_batch = random.sample(replay_memory, batch_size)

            # calculate targets and perform gradient descent
            y_j_batch = get_targets(q_network, target_network, mini_batch, gamma)
            # y_j_batch = np.reshape(y_j_batch, [1, batch_size])

            # update the target network
            q_weights = q_network.get_weights()
            target_weights = target_network.get_weights()
            for i in range(len(target_weights)):
                target_weights[i] = q_weights[i] * tau + target_weights[i] * (1-tau)
            target_network.set_weights(target_weights)

            # target_network.fit(state, y_j_batch, verbose=0)

            state = next_state

            t_count += 1

        print('Reward:', stats.episode_rewards[episode])

    # print end time
    print('End time:', str(datetime.now()))

    return q_network, stats


def test_the_network(env, n_episodes, alpha, gamma, decay, min_epsilon, batch_size, tau,
                     initial_replay_memory_size, replay_memory_size):

    # train the network
    Q_network, stats = deep_q_learning(env, n_episodes=n_episodes, alpha=alpha, gamma=gamma,
                                       decay=decay, min_epsilon=min_epsilon, batch_size=batch_size,
                                       tau=tau, initial_replay_memory_size=initial_replay_memory_size,
                                       replay_memory_size=replay_memory_size)

    # save the weights
    Q_network.save_weights('q_network_weights.h5')

    # plot Reward vs Training Episode
    plt.plot(np.arange(num_episodes), stats.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Training Episode')
    plt.show()

    # test model on 100 trials
    num_trials = 100
    rewards = np.zeros(num_trials)
    for i in range(num_trials):
        # env.render()
        print('Trial:', i)
        state = env.reset()
        done = False

        while not done:
            state = np.reshape(state, [1, np.size(state)])
            action = np.argmax(Q_network.predict(state))
            next_state, reward, done, info = env.step(action)
            rewards[i] += reward
            state = next_state

        print('Reward:', rewards[i])

    # plot 100 trial performance
    plt.plot(np.arange(num_trials), rewards)
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.title('Reward vs 100 Trials')
    plt.text(0, 0, 'learning rate = ' + str(alpha_val))
    plt.show()


def exp_1_alpha_vs_epsilon(env):

    # fixed hyperparameters
    num_episodes = 1000
    gamma_val = 0.999
    min_eps_val = 0.1
    batch_size_val = 32
    tau_val = 0.01
    initial_memory_size_val = 1000
    memory_size_val = 5000

    # experimental hyperparameters
    epsilon_list = [0.0005, 0.0007, 0.001]
    learning_rate_list = [0.01, 0.001, 0.0001]

    avg_rewards = np.zeros((len(learning_rate_list), len(epsilon_list)))

    for r, alpha_val in enumerate(learning_rate_list):
        print('alpha:', alpha_val)
        for c, epsilon_val in enumerate(epsilon_list):
            print('epsilon:', epsilon_val)
            Q_network, stats = deep_q_learning(env, n_episodes=num_episodes, alpha=alpha_val, gamma=gamma_val,
                                               decay=epsilon_val, min_epsilon=min_eps_val, batch_size=batch_size_val,
                                               tau=tau_val, initial_replay_memory_size=initial_memory_size_val,
                                               replay_memory_size=memory_size_val)

            weights_fn = 'weights_' + 'alpha_' + str(alpha_val) + '_epsilon_' + str(epsilon_val) + '.h5'
            Q_network.save_weights(weights_fn)
            avg_reward = np.mean(stats.episode_rewards)
            avg_rewards[r, c] = avg_reward
            print('avg reward:', avg_reward)

    # plot of avg reward vs epsilon decay rate
    plt.plot(epsilon_list, avg_rewards[0, :])
    plt.plot(epsilon_list, avg_rewards[1, :])
    plt.plot(epsilon_list, avg_rewards[2, :])
    plt.xlabel('Epsilon Decay Rate')
    plt.ylabel('Average Reward per 1000 Episodes')
    plt.title('Effect of Learning Rate and Epsilon Decay Rate')
    plt.legend(['learning rate = 0.01', 'learning rate = 0.001', 'learning rate = 0.0001'], loc='upper left')
    plt.show()


### TESTING THE ENVIRONMENT ###
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    env.reset()

    # HYPERPARAMETERS #
    num_episodes = 1000
    alpha_val = 0.0001
    gamma_val = 0.999
    eps_decay_val = 0.001
    min_eps_val = 0.1
    batch_size_val = 32
    tau_val = 0.01
    initial_memory_size_val = 1000
    memory_size_val = 5000

    # Test the network
    test_the_network(env, num_episodes, alpha_val, gamma_val, eps_decay_val, min_eps_val, batch_size_val,
                     tau_val, initial_memory_size_val, memory_size_val)

    # Experiment 1 - learning rate vs epsilon
    # exp_1_alpha_vs_epsilon(env)
