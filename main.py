from dqn_agent import DQNAgent
from constants import Constants as C
from env import ScumEnv
from tqdm import tqdm 
import numpy as np
from utils import print_rl_variables


def main():
    env = ScumEnv(5)
    agents = [DQNAgent(epsilon=C.EPSILON) for _ in range(5)]
    ep_rewards = [[] for _ in range(5)]


    for episode in tqdm(range(1, C.EPISODES + 1), ascii=True, unit='episodes'):
        finish_agents = [False] * 5
        # Restarting episode - reset episode reward and step number
        episode_rewards = [0] * 5
        step = 1

        # Reset environment and get initial state
        env.reset()

        # Reset flag and start iterating until episode ends
        while np.array(finish_agents).sum() != 5:
            for i, agent in enumerate(agents):
                if finish_agents[i] == False:
                    current_state = env.get_cards_to_play()
                    if current_state is None:
                        break
                    action = env.decide_move(current_state, epsilon=agent.epsilon, model=agent)
                    new_state, reward, finish = env.make_move(action)
                    print_rl_variables(reward, new_state, finish, epsilon=agent.epsilon)
                    finish_agents[i] = finish
                    # Update episode reward for the current agent
                    episode_rewards[i] += reward

                    # Every step we update replay memory and train main network
                    agent.update_replay_memory((current_state, action, reward, new_state, finish))
                    agent.train(finish)
                    current_state = new_state
                    step += 1
        
        # Append episode rewards to the lists and log stats
        for i in range(5):
            ep_rewards[i].append(episode_rewards[i])
        if not episode % C.AGGREGATE_STATS_EVERY or episode == 1:
            for i in range(5):
                average_reward = sum(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]) / len(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                min_reward = min(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                max_reward = max(ep_rewards[i][-C.AGGREGATE_STATS_EVERY:])
                print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            
            # Decay epsilon for all agents
            for agent in agents:
                agent.decay_epsilon()
        print("End of episode " + str(episode))
        
if __name__ == "__main__":
    main()

