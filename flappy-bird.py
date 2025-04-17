import argparse
import datetime
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from exprience_replay import ReplayMemory

import torch
import numpy as np
import random
import yaml
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")

class FlappyBird:

    def __init__(self, hyperparam_set):
        with open("hyperparameters.yml", "r") as f:
            all_parameters = yaml.safe_load(f)
            hyperparameters = all_parameters[hyperparam_set]
        
        self.hyperparam_set = hyperparam_set

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.learning_rate = hyperparameters['learning_rate']
        self.discount_factor = hyperparameters['discount_factor']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.fc1_nodes = hyperparameters['fc1_nodes']
        

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        
        # LOG
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_set}.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        # stacks tuples into torch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float()

        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor *  target_dqn(new_states).max(dim=1)[0] 
            # (1-terminations): if the game terminates (1), the rest becomes 0

        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        
        # compute the loss for the whole minibatch
        loss = self.loss_fn(current_q, target_q)

        # optimize the model
        self.optimizer.zero_grad() # clear gradient
        loss.backward() # calculate gradient
        self.optimizer.step() # update parameters


    def play(self, render_mode='human', is_train=True, audio_on=True, use_lidar=False): #, audio_on=True, use_lidar=False - use when flappy bird
        
        if is_train:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

        env = gymnasium.make("FlappyBird-v0", audio_on=audio_on, render_mode=render_mode, use_lidar=use_lidar)
        # env = gymnasium.make("", render_mode="human")

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        reward_history = []
        

        policy_dqn = DQN(state_dim=num_states, action_dim=num_actions, hidden_dim=self.fc1_nodes)

        if is_train:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(state_dim=num_states, action_dim=num_actions, hidden_dim=self.fc1_nodes)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            step_count = 0 # track number of steps taken 
            best_reward = -99999 # track best reward
            epsilon_history = [] # list to keep the history of epsilon decay

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        else: # test
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        for episode in range(1000):
            episode_reward = 0
            terminated = False
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float)

            # Checking if the player is still alive
            while not terminated and episode_reward < self.stop_on_reward:
                # Next action:
                # (feed the observation to your agent here)
                if is_train and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax()

                # Processing:
                new_state, reward, terminated, _, info = env.step(action.item())
                new_state = torch.tensor(new_state, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)
                # print(f"Obs: {obs}\n" f"Score: {info['score']}\n")

                episode_reward += reward

                if is_train:
                    memory.append((state, action, new_state, reward, terminated))
                    step_count += 1

                # update the state
                state = new_state

            reward_history.append(episode_reward) # collect reward history per episode

            if is_train:
                if episode_reward > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as f:
                        f.write(log_message + '\n')

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
            # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=1):
                    self.save_graph(reward_history, epsilon_history)
                    last_graph_update_time = current_time

                # if enough experience has been collected
                if len(memory) > self.mini_batch_size:

                    #sample from memory
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
                    epsilon_history.append(epsilon)

                    # copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0         
            
        env.close()

        # print(state.shape) # (12,) if lidar = False | (180,) if lidar = True

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or Test')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    flappybird = FlappyBird(hyperparam_set=args.hyperparameters)

    if args.train:
        flappybird.play(is_train=True)
    else:
        flappybird.play(is_train=False) # change rendermode to human