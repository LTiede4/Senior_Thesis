"""
Code to train a DQN Agent
Docs: https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
Modified by: Will Solow, 2024
"""

import random
from argparse import Namespace
import time
from dataclasses import dataclass
import wandb

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from rl_algs.rl_utils import RL_Args, Agent, setup, eval_policy

import subprocess
import os
from pathlib import Path
import glob

@dataclass
class Args(RL_Args):
    total_timesteps: int = 5000000   # 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 650
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    checkpoint_frequency: int = 500
    """How often to save the agent during training"""


class DQN(nn.Module, Agent):
    def __init__(self, env: gym.Env, state_fpath: str = None, **kwargs: dict) -> None:
        super().__init__()
        self.env = env
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

        if state_fpath is not None:
            assert isinstance(
                state_fpath, str
            ), f"`state_fpath` must be of type `str` but is of type `{type(state_fpath)}`"
            try:
                self.load_state_dict(torch.load(state_fpath, weights_only=True))
            except:
                msg = f"Error loading state dictionary from {state_fpath}"
                raise Exception(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_action(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Returns action from network. Helps with compatibility
        """
        return torch.argmax(self.network(x), dim=-1)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# runs at periodic iterations
def run_intermediate_scripts(curr_iteration) :
    """
    Find the most recently created folder matching the pattern

    Must change the following variables in this script depending on the test:
     - current_model_name
     - --npk.output-vars
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # CHANGE THIS NAME
    current_model_name = "unconstrained_test"
    pattern = "DQN/grape-lnpkw-v0__rl_utils__1__*"
    folders = glob.glob(os.path.join(base_dir, "data/grapevine", current_model_name, pattern))
    # check if folder exists
    if not folders :
        return None
    dynamic_folder = max(folders, key=os.path.getctime)

    rel_dynamic_folder = os.path.relpath(dynamic_folder, start=base_dir)
    rel_current_iteration_folder = os.path.join(rel_dynamic_folder, str(curr_iteration)) + "/"
    rel_agent_path = os.path.join(rel_dynamic_folder, "agent.pt")
    data_filename = f"{current_model_name}.npz"
    rel_data_file_path = os.path.join(rel_current_iteration_folder, data_filename)

    # Command 1: Generate data
    cmd1 = [
        "python", "-m", "data_generation.gen_data",
        "--save-folder", rel_current_iteration_folder,
        "--data-file", current_model_name,
        "--agent-path", os.path.join(rel_dynamic_folder, "agent.pt"),
        "--agent-type", "DQN",
        "--year-low", "1987", "--year-high", "1987",
        "--lat-low", "43", "--lat-high", "43",
        "--lon-low", "-120", "--lon-high", "-120",
        "--file-type", "npz",
        "--env-id", "grape-lnpkw-v0",
        "--agro-file", "grape_agro.yaml",
        "--npk.output-vars", "LAI", "FIN", "DVS", "WSO", "NAVAIL", "PAVAIL", "KAVAIL", "SM", "TOTN", "TOTP", "TOTK", "TOTIRRIG"
    ]
    # Command 2: Plot output
    cmd2 = [
        "python", "-m", "data_plotting.vis_data",
        "--plt", "plot_output",
        "--data_file", rel_data_file_path,
        "--fig-folder", rel_current_iteration_folder
    ]
    # Command 3: Plot policy
    cmd3 = [
        "python", "-m", "data_plotting.vis_data",
        "--plt", "plot_policy",
        "--data_file", rel_data_file_path,
        "--fig-folder", rel_current_iteration_folder
    ]

    # first make the current iteration directory 
    abs_iteration_folder = os.path.join(dynamic_folder, str(curr_iteration))
    os.makedirs(abs_iteration_folder, exist_ok=True)
    # Run them sequentially
    try:
        print(f"Running scripts for iteration {curr_iteration} in {rel_current_iteration_folder}")
        print("Running data generation...")
        subprocess.run(cmd1, check=True, cwd=base_dir)
        print("Running plot output...")
        subprocess.run(cmd2, check=True, cwd=base_dir)
        print("Running plot policy...")
        subprocess.run(cmd3, check=True, cwd=base_dir)  
    except subprocess.CalledProcessError as e:
        print(f"Error in iteration {curr_iteration}:")
        print(f"Command: {' '.join(e.cmd)}")
        print(f"Error output: {e.stderr}")
        return None


def train(kwargs: Namespace) -> None:
    """
    DQN Training Function
    """
    args = kwargs.alg
    print(f"Starting DQN training for {args.total_timesteps} timesteps...\n")
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"DQN/{kwargs.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer, device, envs = setup(kwargs, args, run_name)

    q_network = DQN(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = DQN(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Track checkpoint count for periodic script execution
    checkpoint_count = 0
    num_eval_runs = 10
    # Calculate how many checkpoints between script runs
    total_checkpoints = args.total_timesteps // args.checkpoint_frequency
    checkpoint_interval = max(1, total_checkpoints // num_eval_runs)

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):

        if global_step % args.checkpoint_frequency == 0:
            torch.save(q_network.state_dict(), f"{kwargs.save_folder}{run_name}/agent.pt")
            if kwargs.track:
                wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

            # Run intermediate scripts at specified checkpoints
            checkpoint_count += 1
            if checkpoint_count % checkpoint_interval == 0:
                current_iteration = global_step // args.train_frequency if args.train_frequency > 0 else global_step
                run_intermediate_scripts(current_iteration)
        
        epsilon = linear_schedule(
            args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step
        )
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if global_step % args.checkpoint_frequency == 0:
            writer.add_scalar("charts/average_reward", eval_policy(q_network, envs, kwargs, device), global_step)

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % args.checkpoint_frequency == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    envs.close()
    writer.close()
