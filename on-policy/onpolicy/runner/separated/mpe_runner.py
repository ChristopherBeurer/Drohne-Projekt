    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
              
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
             
                self.insert(data)

          
            self.compute()
            train_infos = self.train()
            
           
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
          
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

         
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)
    
    def warmup(self):
        obs = self.envs.reset() 
        print(f"[DEBUG] obs is a {type(obs)} of length {len(obs)}")
        print(f"[DEBUG] obs[0] type: {type(obs[0])}, shape: {np.array(obs[0]).shape}")

        for agent_id in range(self.num_agents):
            agent_obs = np.array(obs[agent_id], dtype=np.float32).reshape(1, -1)
            print(f"[DEBUG] assigning buffer[{agent_id}].obs[0] = shape {agent_obs.shape}")
            self.buffer[agent_id].obs[0] = agent_obs

            if hasattr(self.buffer[agent_id], "share_obs"):
                self.buffer[agent_id].share_obs[0] = agent_obs.copy()


    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            print(f"[DEBUG] buffer[agent_id].obs[step] shape: {self.buffer[agent_id].obs[step].shape}")
            print(f"[COLLECT] buffer[{agent_id}].obs[{step}].shape: {self.buffer[agent_id].obs[step].shape}")

            obs = self.buffer[agent_id].obs[step]
            share_obs = self.buffer[agent_id].share_obs[step]

            if obs.ndim == 1:
                obs = obs[None, :]
            if share_obs.ndim == 1:
                share_obs = share_obs[None, :]
            elif obs.ndim > 2:
                raise RuntimeError(f"obs shape too big: {obs.shape}, expected (1, obs_dim) for agent {agent_id}")
            elif share_obs.ndim > 2:
                raise RuntimeError(f"share_obs shape too big: {share_obs.shape}, expected (1, obs_dim) for agent {agent_id}")

            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[agent_id].policy.get_actions(
                share_obs,
                obs,
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step]
            )

            values.append(_t2n(value))
            action = _t2n(action)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

            
            action_space_type = self.envs.action_space[agent_id].__class__.__name__
            if action_space_type == 'Box':
            
                action_env = action      
            elif action_space_type == 'MultiDiscrete':
             
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif action_space_type == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError(f"Unknown action space type: {action_space_type} for agent {agent_id}")

            actions.append(action)
            temp_actions_env.append(action_env)

    
        actions_env = []
        for i in range(self.n_rollout_threads):
            agent_actions = []
            for temp_action_env in temp_actions_env:
                agent_actions.append(temp_action_env[i])
            actions_env.append(agent_actions)

      
        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rewards = np.array(rewards)
        if rewards.ndim == 1:
            rewards = rewards[None, :]


        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = np.array([np.array(o).flatten() for o in obs])

        global_obs = np.concatenate([np.array(o).flatten() for o in obs])  
        global_obs_row = global_obs[None, :]                               

        for agent_id in range(self.num_agents):
            share_obs = np.array([obs[agent_id]]).reshape(1, -1) 
            self.buffer[agent_id].insert(
                share_obs,
                np.array([obs[agent_id]]), 
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id]
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                if isinstance(eval_obs, list):
                  
                    eval_obs_agent = np.array([eval_obs[agent_id]])   
                else:
            
                    eval_obs_agent = np.array(eval_obs)[:, agent_id]
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs_agent,
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True)
                eval_action = eval_action.detach().cpu().numpy()
            
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Box':
                    eval_action_env = eval_action    
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

       
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        print('eval_episode_rewards shape:', eval_episode_rewards.shape) 

        eval_train_infos = []
        for agent_id in range(self.num_agents):
    
            eval_average_episode_rewards = np.mean(eval_episode_rewards[:, agent_id])
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print(f"eval average episode rewards of agent{agent_id}: {eval_average_episode_rewards}")

        self.log_train(eval_train_infos, total_num_steps)


    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
           
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

            
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
