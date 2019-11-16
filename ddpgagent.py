import models
import networks
import tr_helpers
import experience
import tensorflow as tf
import numpy as np
import collections
import time
from collections import deque
from tensorboardX import SummaryWriter


default_config = {
    'GAMMA' : 0.99,
    'ACTOR_LEARNING_RATE' : 1e-3,
    'CRITIC_LEARNING_RATE' : 1e-3,
    'STEPS_PER_EPOCH' : 20,
    'BATCH_SIZE' : 64,
    'EPSILON' : 0.8,
    'EPSILON_DECAY_FRAMES' : 1e5,
    'MIN_EPSILON' : 0.02,
    'NUM_EPOCHS_TO_COPY' : 1000,
    'NUM_STEPS_FILL_BUFFER' : 10000,
    'NAME' : 'DDPG',
    'SCORE_TO_WIN' : 20,
    'REPLAY_BUFFER_TYPE' : 'normal', # 'prioritized'
    'REPLAY_BUFFER_SIZE' :100000,
    'PRIORITY_BETA' : 0.4,
    'PRIORITY_ALPHA' : 0.6,
    'BETA_DECAY_FRAMES' : 1e5,
    'MAX_BETA' : 1.0,
    'NETWORK' : models.ModelDDPG(networks.default_ddpg_network),
    'REWARD_SHAPER' : tr_helpers.DefaultRewardsShaper(),
    'EPISODES_TO_LOG' : 20, 
    'LIVES_REWARD' : 1,
    'STEPS_NUM' : 1,
    }



#(-1, 1) -> (low, high)
def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class OUProcces:
    def __init__(self, n_actions, ou_mu = 0.0, ou_teta=0.15, ou_dt=1, ou_sigma=0.3):
        self.n_actions = n_actions
        self.mu = ou_mu
        self.teta = ou_teta
        self.sigma = ou_sigma
        self.dt = ou_dt
        self.state = np.ones(self.n_actions) * self.mu

    def reset(self):
        self.state = np.ones(self.n_actions) * self.mu

    def get(self, delta):
        sigma = max(0, delta*self.sigma)
        x = self.state + self.teta * (self.mu - self.state) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.n_actions)
        self.state = x
        return x

class DDPGAgent:
    def __init__(self,  env, env_eval, sess, name, observation_space, action_space, config):
        observation_shape = env.observation_space.shape
        self.env_eval = env_eval
        self.env_name = name
        self.actions_low = action_space.low
        self.actions_high = action_space.high
        self.network = config['NETWORK']
        self.config = config
        self.state_shape = observation_shape
        self.action_space = action_space
        self.actions_num = action_space.shape[0]
        self.writer = SummaryWriter()
        self.epsilon = self.config['EPSILON']
        self.rewards_shaper = self.config['REWARD_SHAPER']
        self.epsilon_processor = tr_helpers.LinearValueProcessor(self.config['EPSILON'], self.config['MIN_EPSILON'], self.config['EPSILON_DECAY_FRAMES'])
        self.env = env
        self.sess = sess
        self.steps_num = self.config['STEPS_NUM']
        self.states = deque([], maxlen=self.steps_num)
        self.is_prioritized = config['REPLAY_BUFFER_TYPE'] != 'normal'
        self.atoms_num = self.config['ATOMS_NUM']
        self.is_distributional = self.atoms_num > 1
        self.noise = OUProcces(self.actions_num)
        self.actor_learning_rate = self.config['ACTOR_LEARNING_RATE']
        self.critic_learning_rate = self.config['CRITIC_LEARNING_RATE']
        if self.is_distributional:
            self.v_min = self.config['V_MIN']
            self.v_max = self.config['V_MAX']
            self.delta_z = (self.v_max - self.v_min) /(self.atoms_num - 1)
            self.all_z = tf.range(self.v_min, self.v_max + self.delta_z, self.delta_z)

        if not self.is_prioritized:
            self.exp_buffer = experience.ReplayBuffer(config['REPLAY_BUFFER_SIZE'])
        else: 
            self.beta_processor = tr_helpers.LinearValueProcessor(self.config['PRIORITY_BETA'], self.config['MAX_BETA'], self.config['BETA_DECAY_FRAMES'])
            self.exp_buffer = experience.PrioritizedReplayBuffer(config['REPLAY_BUFFER_SIZE'], config['PRIORITY_ALPHA'])
            self.sample_weights_ph = tf.placeholder(tf.float32, shape= [None,] , name='sample_weights')
        
        self.obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'obs_ph')
        self.actions_ph = tf.placeholder(tf.float32, shape=(None, self.actions_num) , name = 'actions_ph')
        self.a_grads_ph = tf.placeholder(tf.float32, shape=(None, self.actions_num), name = 'grads_ph')
        self.rewards_ph = tf.placeholder(tf.float32, shape=[None,], name = 'rewards_ph')
        self.next_obs_ph = tf.placeholder(tf.float32, shape=(None,) + self.state_shape , name = 'next_obs_ph')
        self.is_done_ph = tf.placeholder(tf.float32, shape=[None,], name = 'is_done_ph')
        self.is_not_done = 1 - self.is_done_ph

        self._reset()
        self.gamma = self.config['GAMMA']
        if self.atoms_num == 1:
            self.setup_qvalues(self.actions_num)
        else:
            self.setup_c51_qvalues(self.actions_num)
        
        self.tau = 0.001
        self.saver = tf.train.Saver()
        self.assigns_hard = [tf.assign(w_target, w_self, validate_shape=True) for w_self, w_target in zip(self.weights, self.target_weights)]
        self.assigns_soft = [tf.assign(w_target, w_self * self.tau + w_target * (1. - self.tau), validate_shape=True) for w_self, w_target in zip(self.weights, self.target_weights)]

        sess.run(tf.global_variables_initializer())
        self.sess.run(self.assigns_hard)

    def init_actor(self):
        self.actor_loss = tf.reduce_mean(-self.qvalues_loss)
        self.actor_train_op = tf.train.AdamOptimizer(self.actor_learning_rate)
        grads = tf.gradients(self.actor_loss, self.actor_weights)
        #grads, _ = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, self.actor_weights))
        self.actor_train_step = self.actor_train_op.apply_gradients(grads)

    def init_critic(self):
        self.gamma_step = tf.constant(self.gamma**self.steps_num, dtype=tf.float32)
        self.reference_qvalues = self.rewards_ph + self.gamma_step * self.is_not_done * tf.stop_gradient(self.target_qvalues)

        self.td_loss_mean = tf.losses.huber_loss(tf.stop_gradient(self.reference_qvalues), self.qvalues)

        self.critic_train_op = tf.train.AdamOptimizer(self.critic_learning_rate)
        grads = tf.gradients(self.td_loss_mean, self.critic_weights)
        #grads, _ = tf.clip_by_global_norm(grads, 0.5)
        grads = list(zip(grads, self.critic_weights))
        self.critic_train_step = self.critic_train_op.apply_gradients(grads)

    def setup_c51_qvalues(self, actions_num):
        self.qvalues_c51 = self.network('agent', self.obs_ph, actions_num)


    def setup_qvalues(self, actions_num):
        agent_dict = {
            'name': 'agent',
            'inputs' : self.obs_ph,
            'actions' : self.actions_ph,
            'actions_num' : actions_num
        }

        agent_loss_dict = {
            'name': 'agent',
            'inputs' : self.obs_ph,
            'actions' : None,
            'actions_num' : actions_num
        }

        target_dict = {
            'name' : 'target',
            'inputs' : self.next_obs_ph,
            'actions' : None,
            'actions_num' : actions_num
        }

        self.actions, self.qvalues = self.network(agent_dict)
        _, self.qvalues_loss = self.network(agent_loss_dict, reuse=True)
        _, self.target_qvalues = self.network(target_dict)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        self.target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        self.critic_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent/critic')
        #self.critic_target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/critic')


        self.actor_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent/actor')
        #self.actor_target_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target/actor')


        self.init_critic()
        self.init_actor()


    def save(self, fn):
        self.saver.save(self.sess, fn)

    def restore(self, fn):
        self.saver.restore(self.sess, fn)

    def _reset(self):
        self.noise.reset()
        self.states.clear()
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.total_shaped_reward = 0.0
        self.step_count = 0

    def get_qvalues(self, state):
        return self.sess.run(self.qvalues, {self.obs_ph: [state]})[0]


    def get_action(self, state):
        actions = self.sess.run(self.actions, {self.obs_ph: [state]})[0] 
        return actions

    def get_action_noise(self, state, epsilon):
        n = self.noise.get(epsilon)
        actions = np.clip(self.sess.run(self.actions, {self.obs_ph: [state]})[0] + n, -1.0, 1.0)
        return actions

    def evaluate(self, env,t_max=1000, n_games = 10):
        rewards = 0
        steps = 0
        for _ in range(n_games):
            s = env.reset()
            reward = 0
            for _ in range(t_max):
                #env.render()
                action = self.get_action(s)
                #print(action)
                s, r, done, _ = env.step(rescale_actions(self.actions_low, self.actions_high, action))
                reward += r
                steps += 1
                    
                if done:
                    rewards += reward
                    break       
            
        return rewards / n_games, steps / n_games

    def play_steps(self, steps, epsilon):
        done_reward = None
        done_shaped_reward = None
        done_steps = None
        steps_rewards = 0
        cur_gamma = 1
        cur_states_len = len(self.states)
        # always break after one
        while True:
            if cur_states_len > 0:
                state = self.states[-1][0]
            else:
                state = self.state
            action = self.get_action_noise(state, epsilon)
            new_state, reward, is_done, _ = self.env.step(rescale_actions(self.actions_low, self.actions_high, action))
            new_state = np.squeeze(new_state)
            reward = reward * (1 - is_done)
 
            self.step_count += 1
            self.total_reward += reward
            shaped_reward = self.rewards_shaper(reward)
            self.total_shaped_reward += shaped_reward
            self.states.append([new_state, action, shaped_reward])

            if len(self.states) < steps:
                break

            for i in range(steps):
                sreward = self.states[i][2]
                steps_rewards += sreward * cur_gamma
                cur_gamma = cur_gamma * self.gamma

            next_state, current_action, _ = self.states[0]
            self.exp_buffer.add(self.state, current_action, steps_rewards, new_state, is_done)
            self.state = next_state
            break

        if is_done:
            done_reward = self.total_reward
            done_steps = self.step_count
            done_shaped_reward = self.total_shaped_reward
            self._reset()
        return done_reward, done_shaped_reward, done_steps


    def load_weigths_into_target_network(self):
        self.sess.run(self.assigns_soft)

    def sample_batch(self, exp_replay, batch_size):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch  = exp_replay.sample(batch_size)
        return {
        self.obs_ph:obs_batch, self.actions_ph : act_batch, self.rewards_ph:reward_batch, 
        self.is_done_ph:is_done_batch, self.next_obs_ph:next_obs_batch
        }, {self.obs_ph:obs_batch}

    def sample_prioritized_batch(self, exp_replay, batch_size, beta):
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch,  sample_weights, sample_idxes = exp_replay.sample(batch_size, beta)
        batch = { self.obs_ph:obs_batch, self.actions_ph : act_batch, self.rewards_ph:reward_batch, 
        self.is_done_ph:is_done_batch, self.next_obs_ph:next_obs_batch, self.sample_weights_ph: sample_weights }
        return [batch , sample_idxes]

    def train(self):
        self.beta = 0
        last_mean_rewards = -100500
        for _ in range(0, self.config['NUM_STEPS_FILL_BUFFER']):
            self.play_steps(self.steps_num, self.epsilon)

        STEPS_PER_EPOCH = self.config['STEPS_PER_EPOCH']
        BATCH_SIZE = self.config['BATCH_SIZE']
        LIVES_REWARD = self.config['LIVES_REWARD']
        EPISODES_TO_LOG = self.config['EPISODES_TO_LOG']
        FRAMES_TO_LOG = self.config['FRAMES_TO_LOG']
        frame = 0
        play_time = 0
        update_time = 0
        rewards = []
        shaped_rewards = []
        steps = []
        c_losses = 0
        a_losses = 0
        while True:
            t_play_start = time.time()
            self.epsilon = self.epsilon_processor(frame)
            if self.is_prioritized:
                self.beta = self.beta_processor(frame)

            for _ in range(0, STEPS_PER_EPOCH):
                reward, shaped_reward, step = self.play_steps(self.steps_num, self.epsilon)
                if reward != None:
                    steps.append(step)
                    rewards.append(reward)
                    shaped_rewards.append(shaped_reward)

            t_play_end = time.time()
            play_time += t_play_end - t_play_start
            
            # train
            frame = frame + STEPS_PER_EPOCH
            
            t_start = time.time()
          
            batch, actor_batch = self.sample_batch(self.exp_buffer, batch_size=BATCH_SIZE)

            _, c_loss = self.sess.run([self.critic_train_step, self.td_loss_mean], batch)
            _, a_loss = self.sess.run([self.actor_train_step, self.actor_loss], actor_batch)
            c_losses += c_loss
            a_losses += a_loss
            self.load_weigths_into_target_network()

            t_end = time.time()
            update_time += t_end - t_start



            if frame % 1024 == 0:
                print('Frames per seconds: ', 1024 / (update_time + play_time))
                self.writer.add_scalar('Frames per seconds: ', 1024 / (update_time + play_time), frame)
                self.writer.add_scalar('upd_time', update_time, frame)
                self.writer.add_scalar('play_time', play_time, frame)
                self.writer.add_scalar('critic_loss', c_losses / 1024.0, frame)
                self.writer.add_scalar('actor_loss', a_losses / 1024.0, frame)
                self.writer.add_scalar('epsilon', self.epsilon, frame)
                c_losses = 0
                a_losses = 0

                if self.is_prioritized:
                    self.writer.add_scalar('beta', self.beta, frame)
                    
                update_time = 0
                play_time = 0
            
            if frame % FRAMES_TO_LOG == 0:
                mean_reward, mean_steps = self.evaluate(self.env_eval, n_games = EPISODES_TO_LOG)
                print('mean_rewards: ', mean_reward, 'mean_steps: ', mean_steps)
                if mean_reward > last_mean_rewards:
                    print('saving next best rewards: ', mean_reward)
                    last_mean_rewards = mean_reward
                    self.save("./nn/" + self.config['NAME'] + self.env_name)
                    if last_mean_rewards > self.config['SCORE_TO_WIN']:
                        print('Network won!')
                        return

                self.writer.add_scalar('steps', mean_steps, frame)
                self.writer.add_scalar('reward', mean_reward, frame)                
                
                
