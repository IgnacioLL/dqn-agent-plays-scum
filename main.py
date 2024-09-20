from dqn_agent import AgentPool
from constants import Constants as C
from env import ScumEnv
from tqdm import tqdm 
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

def run_episode(env, agent_pool, total_steps) -> tuple[list[int], int]:
    finish_agents = [False] * 5
    episode_rewards = [0] * 5
    env.reset()
    print("Starting episode")
    print("_" * 100)

    while np.array(finish_agents).sum() != agent_pool.number_of_agents:
        print("_"*4)
        agent = agent_pool.get_agent(env.player_turn)
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
        print("_"*4)

    return episode_rewards, total_steps

def log_stats(agent_pool: AgentPool, ep_rewards: list[list[int]], episode: int):
        for i in range(agent_pool.number_of_agents):
            recent_rewards = ep_rewards[i][-C.AGGREGATE_STATS_EVERY:]
            average_reward = sum(recent_rewards) / len(recent_rewards)
            min_reward = min(recent_rewards)
            max_reward = max(recent_rewards)
            print(f"Agent {i+1}: Avg: {average_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f} with epsilon {agent_pool.get_agent(i).epsilon} and learning rate {agent_pool.get_agent(i).learning_rate}")
            writer.add_scalar(f"Reward/Agent {i+1}/Avg Reward", average_reward, episode)
            writer.flush()
            yield i, average_reward

def save_models(agent_pool: AgentPool, i: int) -> None:
    agent_pool.get_agent(i).save_model(path=f"models/checkpoints/agent_{i+1}.pt")

def main(load_checkpoints: bool = False, number_of_agents: int = 5, lr: float = 1e-4) -> None:
    env = ScumEnv(number_of_agents)
    agent_pool = AgentPool(number_of_agents, load_checkpoints=load_checkpoints, learning_rate=lr)
    ep_rewards = [[] for _ in range(number_of_agents)]
    total_steps = 0

    for episode in tqdm(range(1, C.EPISODES + 1), ascii=True, unit='episodes'):
        episode_rewards, total_steps = run_episode(env, agent_pool, total_steps)

        for i, reward in enumerate(episode_rewards):
            ep_rewards[i].append(reward)

        if episode % C.AGGREGATE_STATS_EVERY == 0:
            average_rewards = []
            for i, average_reward in log_stats(agent_pool, ep_rewards, episode):
                average_rewards.append(average_reward)
                save_models(agent_pool, i)
    
            agent_pool.refresh_agents()
            agent_pool.update_order(average_rewards)
            agent_pool.save_agents()


        for agent in range(agent_pool.number_of_agents):
            agent_pool.get_agent(agent).decay_epsilon()

if __name__ == "__main__":
    main(number_of_agents=C.NUMBER_OF_AGENTS, lr=1e-4)