import common.a2c_common
from torch import optim
import torch 
from torch import nn
import algos_torch.torch_ext
import numpy as np

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

class A2CAgent(common.a2c_common.ContinuousA2CBase):
    def __init__(self, base_name, observation_space, action_space, config):
        common.a2c_common.ContinuousA2CBase.__init__(self, base_name, observation_space, action_space, config, False)
        
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : self.state_shape,
            'games_num' : 1,
            'batch_num' : 1,
        } 
        self.model = self.network.build(config)
        self.model.cuda()
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr))
        #self.optimizer = algos_torch.torch_ext.RangerQH(self.model.parameters(), float(self.last_lr))
        batch_size = self.num_agents * self.num_actors
        if observation_space.dtype == np.uint8:
            torch_dtype = torch.uint8
        else:
            torch_dtype = torch.float32

        self.mb_obs = torch.zeros((self.steps_num, batch_size) + self.state_shape, dtype=torch_dtype).cuda()
        self.mb_rewards = torch.zeros((self.steps_num, batch_size), dtype = torch.float32).cuda()
        self.mb_actions = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()
        self.mb_values = torch.zeros((self.steps_num, batch_size), dtype = torch.float32).cuda()
        self.mb_dones = torch.zeros((self.steps_num, batch_size), dtype = torch.long).cuda()
        self.mb_neglogpacs = torch.zeros((self.steps_num, batch_size), dtype = torch.float32).cuda()
        self.mb_mus = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()
        self.mb_sigmas = torch.zeros((self.steps_num, batch_size, self.actions_num), dtype = torch.float32).cuda()

    def play_steps(self):
        # Here, we init the lists that will contain the mb of experiences
        #mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs, mb_mus, mb_sigmas = [],[],[],[],[],[],[],[]
        mb_states = []
        epinfos = []
        #'''
        mb_obs = self.mb_obs
        mb_rewards = self.mb_rewards
        mb_actions = self.mb_actions
        mb_values = self.mb_values
        mb_neglogpacs = self.mb_neglogpacs
        mb_mus = self.mb_mus
        mb_sigmas = self.mb_sigmas
        mb_dones = self.mb_dones
        #'''
        #mb_states = self.mb_states
        # For n in range number of steps
        for n in range(self.steps_num):
            if self.network.is_rnn():
                mb_states.append(self.states)

            self.obs = self._preproc_obs(self.obs)
            actions, values, neglogpacs, mu, sigma, self.states = self.get_action_values(self.obs)
            values = np.squeeze(values)
            neglogpacs = np.squeeze(neglogpacs)
     
            
            mb_obs[n,:] = self.obs
            mb_dones[n,:] = torch.cuda.LongTensor(self.dones)

            self.obs, rewards, self.dones, infos = self.vec_env.step(common.a2c_common.rescale_actions(self.actions_low, self.actions_high, np.clip(actions.cpu().numpy(), -1.0, 1.0)))
            self.current_rewards += rewards
            self.current_lengths += 1
            for reward, length, done in zip(self.current_rewards, self.current_lengths, self.dones):
                if done:
                    self.game_rewards.append(reward)
                    self.game_lengths.append(length)

            shaped_rewards = self.rewards_shaper(rewards)
            epinfos.append(infos)

            mb_actions[n,:] = actions
            mb_values[n,:] = values
            mb_neglogpacs[n,:] = neglogpacs
            
            mb_mus[n,:] = mu
            mb_sigmas[n,:] = sigma
            mb_rewards[n,:] = torch.cuda.FloatTensor(shaped_rewards)
            
            self.current_rewards = self.current_rewards * (1.0 - self.dones)
            self.current_lengths = self.current_lengths * (1.0 - self.dones)
        self.obs = self._preproc_obs(self.obs)
        last_values = self.get_values(self.obs)
        last_values = last_values.squeeze()

        mb_returns = torch.zeros_like(mb_rewards)
        mb_advs = torch.zeros_like(mb_rewards)
        lastgaelam = 0
        
        for t in reversed(range(self.steps_num)):
            if t == self.steps_num - 1:
                nextnonterminal = torch.cuda.FloatTensor(1.0 - self.dones)
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal  - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam

        mb_returns = mb_advs + mb_values

        if self.network.is_rnn():
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_mus, mb_sigmas, mb_states )), epinfos)
        else:
            result = (*map(swap_and_flatten01, (mb_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, mb_mus, mb_sigmas)), None, epinfos)

        return result

    def update_epoch(self):
        self.epoch_num += 1
        return self.epoch_num

    def _tensor_from_obs(self, obs):
        if type(obs) is not np.ndarray:
            return obs
        if obs.dtype == np.uint8:
            obs = torch.cuda.ByteTensor(obs)
        else:
            obs = torch.cuda.FloatTensor(obs)    
        return obs    

    def _preproc_obs(self, obs_batch):
        obs_batch = self._tensor_from_obs(obs_batch)
        if obs_batch.dtype == np.uint8:
            obs_batch = obs_batch.float() / 255.0

        if len(obs_batch.size()) == 4:
            obs_batch = obs_batch.permute((0, 3, 1, 2))
        return obs_batch

    def save(self, fn):
        algos_torch.torch_ext.save_scheckpoint(fn, self.epoch_num, self.model, self.optimizer)

    def restore(self, fn):
        self.epoch_num = algos_torch.torch_ext.load_checkpoint(fn, self.model, self.optimizer)

    def get_masked_action_values(self, obs, action_masks):
        assert False


    def get_action_values(self, obs):
        #obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs,
        }
        with torch.no_grad():
            neglogp, value, action, mu, sigma = self.model(input_dict)
        return action.detach(), \
                value.detach(), \
                neglogp.detach(), \
                mu.detach(), \
                sigma.detach(), \
                None

    def get_values(self, obs):
        #obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'inputs' : obs
        }
        with torch.no_grad():
            neglogp, value, action, mu, sigma = self.model(input_dict)
        return value.detach()

    def get_weights(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())
    
    def set_weights(self, weights):
        torch.nn.utils.vector_to_parameters(weights, self.model.parameters())

    def train_actor_critic(self, input_dict):
        self.model.train()
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        #obs_batch = self._preproc_obs(obs_batch)
        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        input_dict = {
            'is_train': True,
            'prev_actions': actions_batch, 
            'inputs' : obs_batch
        }
        action_log_probs, values, entropy, mu, sigma = self.model(input_dict)
        if self.ppo:
            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - curr_e_clip,
                                1.0 + curr_e_clip) * advantage
            a_loss = torch.max(-surr1, -surr2).mean()
        else:
            a_loss = (action_log_probs * advantage).mean()

        values = torch.squeeze(values)
        if self.clip_value:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-curr_e_clip, curr_e_clip)
            value_losses = (values - return_batch)**2
            value_losses_clipped = (value_pred_clipped - return_batch)**2
            c_loss = torch.max(value_losses,
                                         value_losses_clipped)
        else:
            c_loss = (return_batch - values)**2
        
        c_loss = c_loss.mean()
        if self.bounds_loss_coef is not None:
            soft_bound = 1.1
            mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0)**2
            mu_loss_low = torch.clamp_max(- mu + soft_bound, 0.0)**2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1).mean()
        else:
            b_loss = 0
        loss = a_loss + 0.5 *c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            kl_dist = algos_torch.torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch)
            kl_dist = kl_dist.item()
            if self.is_adaptive_lr:
                if kl_dist > (2.0 * self.lr_threshold):
                    self.last_lr = max(self.last_lr / 1.5, 1e-6)
                if kl_dist < (0.5 * self.lr_threshold):
                    self.last_lr = min(self.last_lr * 1.5, 1e-2)        
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.last_lr

        return a_loss.item(), c_loss.item(), entropy.item(), \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss.item()
