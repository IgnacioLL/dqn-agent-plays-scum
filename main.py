from dqn_agent import DQNAgent
from constants import Constants as C
from env import ScumEnv
from tqdm import tqdm 
import numpy as np
from utils import print_rl_variables
import torch

def main():
    env = ScumEnv(5)
    agents = [DQNAgent(epsilon=C.EPSILON, learning_rate=0.0001) for _ in range(5)]
    ep_rewards = [[] for _ in range(5)]


    for episode in tqdm(range(1, C.EPISODES + 1), ascii=True, unit='episodes'):
        finish_agents = [False] * 5
        # Restarting episode - reset episode reward and step number
        episode_rewards = [0] * 5
        step = 1

        # Reset environment and get initial state
        env.reset() 
        print("Starting episode")
        print("_"*100)
        # Reset flag and start iterating until episode ends
        while np.array(finish_agents).sum() != 5:

            agent = agents[env.player_turn]
            action_state = env.get_cards_to_play()
            action = env.decide_move(action_state, epsilon=agent.epsilon, agent=agent)
            env._print_move(action)
            current_state, new_state, reward, finish, agent_number = env.make_move(action)
            finish_agents[agent_number] = finish

            episode_rewards[agent_number] += reward

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, False))
            
            if finish:
                current_state, new_state, reward = env.get_stats_after_finish(agent_number=agent_number)
                episode_rewards[agent_number] += reward
                agent.update_replay_memory((current_state, action, reward, new_state, finish))
                finish_agents[agent_number] = finish

            agent.train(finish)
            step += 1
            print("Number of players finished: ", finish_agents)

        # Append episode rewards to the lists and log stats
        for i in range(5):
            ep_rewards[i].append(episode_rewards[i])
        if not episode % C.AGGREGATE_STATS_EVERY or episode == 1:
            for i in range(5):
                average_reward = sum(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]) / len(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} with epsilon {agents[i].epsilon} and learning rate {agents[i].learning_rate}")

        # Decay epsilon for all agents
        # For now we will only train the first agent if it works, then we will train the others
        agents[0].decay_epsilon()
        
if __name__ == "__main__":
    main()

