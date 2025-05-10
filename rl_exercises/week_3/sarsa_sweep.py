"""Run multiple SARSA episodes using Hydra-configured components.

This script uses Hydra to instantiate the environment, policy, and SARSA agent from config files,
then runs multiple episodes and returns the average total reward.
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


def run_episodes(agent, env, num_episodes=5, discount_factor=1.0):
    """Run multiple episodes using the SARSA algorithm.

    Each episode is executed with the agent's current policy. The agent updates its Q-values
    after every step using the SARSA update rule.

    Parameters
    ----------
    agent : object
        An agent implementing `predict_action` and `update_agent`.
    env : gym.Env
        The environment in which the agent interacts.
    num_episodes : int, optional
        Number of episodes to run, by default 5.
    discount_factor : float, optional
        Discount factor for future rewards, by default 1.0.

    Returns
    -------
    float
        Mean total discounted reward across all episodes.
    """
    total_discounted_rewards = []

    for episode in range(num_episodes):
        total_reward = 0.0
        state, _ = env.reset()
        done = False
        action = agent.predict_action(state)
        step = 0

        while not done:
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            next_action = agent.predict_action(next_state)
            agent.update_agent(state, action, reward, next_state, next_action, done)

            # Apply discount factor to the reward
            total_reward += (discount_factor**step) * reward
            state, action = next_state, next_action
            step += 1

        total_discounted_rewards.append(total_reward)

    # Return the mean discounted reward across episodes
    return sum(total_discounted_rewards) / len(total_discounted_rewards)


# Decorate the function with the path of the config file and the particular config to use
@hydra.main(
    config_path="../configs/agent/", config_name="sarsa_sweep", version_base="1.1"
)
def main(cfg: DictConfig) -> dict:
    """Main function to run SARSA with Hydra-configured components.

    This function sets up the environment, policy, and agent using Hydra-based
    configuration, seeds them for reproducibility, and runs multiple episodes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing `env`, `policy`, `agent`, and optionally `seed`.

    Returns
    -------
    float
        Mean total reward across the episodes.
    """

    # Hydra-instantiate the env
    env = instantiate(cfg.env)
    # instantiate the policy (passing in env!)
    policy = instantiate(cfg.policy, env=env)
    # 3) instantiate the agent (passing in env & policy)
    agent = instantiate(cfg.agent, env=env, policy=policy)

    # 4) (optional) reseed for reproducibility
    if cfg.seed is not None:
        env.reset(seed=cfg.seed)
        env.action_space.seed(cfg.seed)

    # 5) run & return reward
    total_reward = run_episodes(agent, env, cfg.num_episodes)
    return total_reward


if __name__ == "__main__":
    main()
