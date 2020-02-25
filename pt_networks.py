import numpy as np

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

#import tensorflow as tf
#import tensorflow_probability as tfp
#tfd = tfp.distributions

def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def sample_noise(shape, mean = 0.0, std = 1.0):
    noise = tf.random_normal(shape, mean = mean, stddev = std)
    return noise


def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def default_small_a2c_network_separated(name, inputs, actions_num, continuous=False, reuse=False, activation=tf.nn.elu):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 128
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 32
        
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, kernel_initializer=normc_initializer(0.01), activation=None)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def default_a2c_network_separated(name, inputs, actions_num, continuous=False, reuse=False, activation=tf.nn.elu):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, kernel_initializer=normc_initializer(1.0), activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, kernel_initializer=normc_initializer(1.0), activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None, kernel_initializer=hidden_init)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, kernel_initializer=normc_initializer(0.01), activation=None)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def default_a2c_network_separated_logstd(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        hidden_init = normc_initializer(1.0) # tf.random_normal_initializer(stddev= 1.0)
        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=tf.nn.elu, kernel_initializer=hidden_init)

        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu, kernel_initializer=hidden_init)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=tf.nn.elu, kernel_initializer=hidden_init)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None, kernel_initializer=hidden_init)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None,)
            #std = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            #logstd = tf.layers.dense(inputs=hidden2a, units=actions_num)
            logstd = tf.get_variable(name='log_std', shape=(actions_num), initializer=tf.constant_initializer(0.0), trainable=True)
            return mu, mu * 0 + logstd, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value


def default_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64

        hidden0 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.relu)
        hidden1 = tf.layers.dense(inputs=hidden0, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        value = tf.layers.dense(inputs=hidden2, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2, units=actions_num, activation=None)
            return logits, value

def default_a2c_lstm_network(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 128
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 64
        LSTM_UNITS = 64
        hidden0 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.relu)
        hidden1 = tf.layers.dense(inputs=hidden0, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        dones_ph = tf.placeholder(tf.float32, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state


def default_a2c_lstm_network_separated(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES0 = 256
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64
        LSTM_UNITS = 128

        hidden0c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu)
        hidden1c = tf.layers.dense(inputs=hidden0c, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu)

        hidden0a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES0, activation=tf.nn.elu)
        hidden1a = tf.layers.dense(inputs=hidden0a, units=NUM_HIDDEN_NODES1, activation=tf.nn.elu)

        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        hidden = tf.concat((hidden1a, hidden1c), axis=1)
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_a', hidden, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        lstm_outa, lstm_outc = tf.split(lstm_out, 2, axis=1)

        value = tf.layers.dense(inputs=lstm_outc, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01))
            var = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state

def simple_a2c_lstm_network_separated(name, inputs, actions_num, games_num, batch_num, continuous=False, reuse=False):
    env_num = games_num

    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 32
        NUM_HIDDEN_NODES2 = 32
        #NUM_HIDDEN_NODES3 = 16
        LSTM_UNITS = 16

        hidden1c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        hidden1a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)


        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2* 2*LSTM_UNITS])
        states_a, states_c = tf.split(states_ph, 2, axis=1)
        lstm_outa, lstm_statae, initial_statea = openai_lstm('lstm_actions', hidden2a, dones_ph=dones_ph, states_ph=states_a, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)

        lstm_outc, lstm_statec, initial_statec = openai_lstm('lstm_critics', hidden2c, dones_ph=dones_ph, states_ph=states_c, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        initial_state = np.concatenate((initial_statea, initial_statec), axis=1)
        lstm_state = tf.concat( values=(lstm_statae, lstm_statec), axis=1)
        #lstm_outa = tf.layers.dense(inputs=lstm_outa, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        #lstm_outc = tf.layers.dense(inputs=lstm_outc, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)
        value = tf.layers.dense(inputs=lstm_outc, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_outa, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state

def simple_a2c_lstm_network(name, inputs, actions_num, env_num, batch_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 32
        NUM_HIDDEN_NODES2 = 32
        LSTM_UNITS = 16
        hidden1 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        dones_ph = tf.placeholder(tf.bool, [batch_num])
        states_ph = tf.placeholder(tf.float32, [env_num, 2*LSTM_UNITS])
        lstm_out, lstm_state, initial_state = openai_lstm('lstm_ac', hidden2, dones_ph=dones_ph, states_ph=states_ph, units=LSTM_UNITS, env_num=env_num, batch_num=batch_num)
        value = tf.layers.dense(inputs=lstm_out, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value, states_ph, dones_ph, lstm_state, initial_state
        else:
            logits = tf.layers.dense(inputs=lstm_out, units=actions_num, activation=None)
            return logits, value, states_ph, dones_ph, lstm_state, initial_state

def simple_a2c_network_separated(name, inputs, actions_num, activation = tf.nn.elu, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 64
        NUM_HIDDEN_NODES2 = 64
        
        hidden1c = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=activation)
        hidden2c = tf.layers.dense(inputs=hidden1c, units=NUM_HIDDEN_NODES2, activation=activation)

        hidden1a = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=activation)
        hidden2a = tf.layers.dense(inputs=hidden1a, units=NUM_HIDDEN_NODES2, activation=activation)

        value = tf.layers.dense(inputs=hidden2c, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2a, units=actions_num, activation=None)
            return logits, value

def simple_a2c_network(name, inputs, actions_num, continuous=False, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        NUM_HIDDEN_NODES1 = 128
        NUM_HIDDEN_NODES2 = 64

        hidden1 = tf.layers.dense(inputs=inputs, units=NUM_HIDDEN_NODES1, activation=tf.nn.relu)
        hidden2 = tf.layers.dense(inputs=hidden1, units=NUM_HIDDEN_NODES2, activation=tf.nn.relu)

        value = tf.layers.dense(inputs=hidden2, units=1, activation=None)
        if continuous:
            mu = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.tanh)
            var = tf.layers.dense(inputs=hidden2, units=actions_num, activation=tf.nn.softplus)
            return mu, var, value
        else:
            logits = tf.layers.dense(inputs=hidden2, units=actions_num, activation=None)
            return logits, value


