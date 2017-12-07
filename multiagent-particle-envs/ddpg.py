import numpy as np
import tensorflow as tf
import pathlib
import general_utilities


class Actor:
    def __init__(self, scope, session, n_actions, action_bound,
                 eval_states, target_states, learning_rate=0.001, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.action_bound = action_bound
        self.eval_states = eval_states
        self.target_states = target_states
        self.learning_rate = learning_rate

        with tf.variable_scope(scope):
            self.eval_actions = self.build_network(self.eval_states,
                                                   scope='eval', trainable=True)
            self.target_actions = self.build_network(self.target_states,
                                                     scope='target', trainable=False)

        self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=scope + '/eval')
        self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope=scope + '/target')

        self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                              for t, e in zip(self.target_weights, self.eval_weights)]

        self.saver = tf.train.Saver()

    def build_network(self, x, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)
            h1 = tf.layers.dense(x, 50, activation=tf.nn.relu,
                                 kernel_initializer=W, bias_initializer=b,
                                 name='h1', trainable=trainable)
            h2 = tf.layers.dense(h1, 50, activation=tf.nn.relu,
                                 kernel_initializer=W, bias_initializer=b,
                                 name='h2', trainable=trainable)
            actions = tf.layers.dense(h2, self.n_actions, activation=tf.nn.tanh,
                                      kernel_initializer=W, bias_initializer=b,
                                      name='actions', trainable=trainable)
            scaled_actions = tf.multiply(actions, self.action_bound,
                                         name='scaled_actions')

        return scaled_actions

    def add_gradients(self, action_gradients):
        self.action_gradients = tf.gradients(ys=self.eval_actions,
                                             xs=self.eval_weights,
                                             grad_ys=action_gradients)
        optimizer = tf.train.AdamOptimizer(-self.learning_rate)
        self.optimize = optimizer.apply_gradients(zip(self.action_gradients,
                                                      self.eval_weights))

    def learn(self, states):
        self.session.run(self.optimize, feed_dict={self.eval_states: states})
        self.session.run(self.update_target)

    def choose_action(self, state):
        return self.session.run(self.eval_actions,
                                feed_dict={self.eval_states: state[np.newaxis, :]})[0]

    def load(self, name):
        latest_ckpt = tf.train.latest_checkpoint(name)
        if latest_ckpt:
            print("Loading model from checkpoint {}".format(latest_ckpt))
            self.saver.restore(self.session, latest_ckpt)

    def save(self, name):
        p = pathlib.Path(name)
        if len(p.parts) > 1:
            dump_dirs = pathlib.Path(*p.parts[:-1])
            general_utilities.ensure_directory_exists(str(dump_dirs))
        save_path = self.saver.save(self.session, name)


class Critic:
    def __init__(self, scope, session, n_actions, actor_eval_actions,
                 actor_target_actions, eval_states, target_states,
                 rewards, learning_rate=0.001, gamma=0.9, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.actor_eval_actions = actor_eval_actions
        self.actor_target_actions = actor_target_actions
        self.eval_states = eval_states
        self.target_states = target_states
        self.rewards = rewards

        with tf.variable_scope(scope):
            self.eval_values = self.build_network(self.eval_states,
                                                  self.actor_eval_actions,
                                                  'eval', trainable=True)
            self.target_values = self.build_network(self.target_states,
                                                    self.actor_target_actions,
                                                    'target', trainable=False)

        self.eval_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=scope + '/eval')
        self.target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope=scope + '/target')

        self.target = self.rewards + gamma * self.target_values
        self.loss = tf.reduce_mean(tf.squared_difference(self.target,
                                                         self.eval_values))

        self.optimize = tf.train.AdamOptimizer(
            learning_rate).minimize(self.loss)
        self.action_gradients = tf.gradients(ys=self.eval_values,
                                             xs=self.actor_eval_actions)[0]

        self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                              for t, e in zip(self.target_weights, self.eval_weights)]

        self.saver = tf.train.Saver()

    def build_network(self, x1, x2, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)
            h1 = tf.layers.dense(x1, 50, activation=tf.nn.relu,
                                 kernel_initializer=W, bias_initializer=b,
                                 name='h1', trainable=trainable)
            h21 = tf.get_variable('h21', [50, 50],
                                  initializer=W, trainable=trainable)
            h22 = tf.get_variable('h22', [self.n_actions, 50],
                                  initializer=W, trainable=trainable)
            b2 = tf.get_variable('b2', [1, 50],
                                 initializer=b, trainable=trainable)
            h3 = tf.nn.relu(tf.matmul(h1, h21) + tf.matmul(x2, h22) + b2)
            values = tf.layers.dense(h3, 1, kernel_initializer=W,
                                     bias_initializer=b, name='values',
                                     trainable=trainable)

        return values

    def learn(self, states, actions, rewards, states_next):
        self.session.run(self.optimize, feed_dict={self.eval_states: states,
                                                   self.actor_eval_actions: actions,
                                                   self.rewards: rewards,
                                                   self.target_states: states_next})
        self.session.run(self.update_target)

    def load(self, name):
        latest_ckpt = tf.train.latest_checkpoint(name)
        if latest_ckpt:
            print("Loading model from checkpoint {}".format(latest_ckpt))
            self.saver.restore(self.session, latest_ckpt)

    def save(self, name):
        p = pathlib.Path(name)
        if len(p.parts) > 1:
            dump_dirs = pathlib.Path(*p.parts[:-1])
            general_utilities.ensure_directory_exists(str(dump_dirs))
        save_path = self.saver.save(self.session, name)
