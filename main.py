from dqn_agent import DQNAgent
from constants import Constants as C
from env import ScumEnv
from tqdm import tqdm 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl

writer = SummaryWriter()

def initialize_agents(load_checkpoints):
    if load_checkpoints:
        return [DQNAgent(epsilon=C.EPSILON, learning_rate=1e-4, path=f"models/checkpoints/agent_{i+1}.pt") for i in range(5)]
    return [DQNAgent(epsilon=C.EPSILON, learning_rate=1e-4) for _ in range(5)]

def run_episode(env, agents, ep_rewards, total_steps):
    finish_agents = [False] * 5
    episode_rewards = [0] * 5
    env.reset()
    print("Starting episode")
    print("_" * 100)

    while np.array(finish_agents).sum() != 5:
        agent = agents[env.player_turn]
        action_state = env.get_cards_to_play()
        action = env.decide_move(action_state, epsilon=agent.epsilon, agent=agent)
        env._print_move(action)
        
        current_state, new_state, reward, finish, agent_number = env.make_move(action)
        finish_agents[agent_number] = finish
        episode_rewards[agent_number] += reward

        agent.update_replay_memory((current_state, action, reward, new_state, False))
        
        if finish:
            current_state, new_state, reward = env.get_stats_after_finish(agent_number=agent_number)
            episode_rewards[agent_number] += reward
            agent.update_replay_memory((current_state, action, reward, new_state, finish))
            finish_agents[agent_number] = finish

        td_error, tree_idxs = agent.train(agent_number, total_steps, finish)
        if td_error is not None:
            agent.buffer.update_priorities(tree_idxs, td_error)  
        total_steps += 1

    return episode_rewards, total_steps

def log_stats(agents, ep_rewards, episode):
    if episode % C.AGGREGATE_STATS_EVERY == 0:
        for i in range(5):
            recent_rewards = ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} with epsilon {agents[i].epsilon} and learning rate {agents[i].learning_rate}")
            writer.add_scalar(f"Reward/Agent {i+1}/Avg Reward", average_reward, episode)
            writer.flush()
            yield i, average_reward

def save_models(agents, max_average_reward, i, average_reward, episode):
    if average_reward > max_average_reward[i]:
        max_average_reward[i] = average_reward
        agents[i].save_model(path=f"models/best_models/agent_{i+1}_episode_{episode}_max_avg_{max_average_reward[i]:.2f}.pt")
    else:
        agents[i].save_model(path=f"models/checkpoints/agent_{i+1}.pt")

def main(load_checkpoints: bool = False):
    env = ScumEnv(5)
    agents = initialize_agents(load_checkpoints)
    ep_rewards = [[] for _ in range(5)]
    max_average_reward = [-np.inf for _ in range(5)]
    total_steps = 0

    for episode in tqdm(range(1, C.EPISODES + 1), ascii=True, unit='episodes'):
        episode_rewards, total_steps = run_episode(env, agents, ep_rewards, total_steps)

        for i, reward in enumerate(episode_rewards):
            if reward < -20:
                print("The reward was lower than -20 check what happened")
                assert False, "The reward was lower than -20 check what happened"
            ep_rewards[i].append(reward)

        for i, average_reward in log_stats(agents, ep_rewards, episode):
            save_models(agents, max_average_reward, i, average_reward, episode)

        for agent in agents[:4]:
            agent.decay_epsilon()


        for i in range(5):
            pkl.dump(agents[i].buffer, open(f"buffer_{i}.pkl", "wb"))

if __name__ == "__main__":
    main()