import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt

def test_agent(env, weights_fn, learning_rate):

    # initial params
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # initialize the agent
    test_agent = Sequential()
    test_agent.add(Dense(units=24, activation='relu', input_dim=state_size))
    test_agent.add(Dense(units=24, activation='relu'))
    test_agent.add(Dense(action_size, activation='linear'))
    test_agent.compile(loss='mse', optimizer=Adam(lr=learning_rate))

    # load the weights
    test_agent.load_weights(weights_fn)

    # perform 100 trials
    num_trials = 100
    rewards = np.zeros(num_trials)
    for i in range(num_trials):
        env.render()
        print('Trial:', i)
        state = env.reset()
        done = False

        while not done:
            state = np.reshape(state, [1, np.size(state)])
            action = np.argmax(test_agent.predict(state))
            next_state, reward, done, info = env.step(action)
            env.render()
            rewards[i] += reward
            state = next_state

        print('Reward:', rewards[i])

    # plot 100 trial performance
    plt.plot(np.arange(num_trials), rewards)
    plt.xlabel('Trial')
    plt.ylabel('Reward')
    plt.title('Reward vs 100 Trials')
    plt.text(0, 0, 'learning rate = ' + str(learning_rate))
    plt.show()


if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    # env = gym.make('CartPole-v0')
    env.reset()

    weights_fn = 'weights_alpha_0.0001_epsilon_0.001.h5'
    learning_rate = 0.0001

    test_agent(env, weights_fn, learning_rate)
