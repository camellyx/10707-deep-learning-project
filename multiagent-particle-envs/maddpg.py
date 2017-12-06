import numpy as np
import tensorflow as tf


class Critic:
    def __init__(self, scope, session, n_actions, actors_eval_actions,
                 actors_target_actions, eval_states, target_states,
                 rewards, learning_rate=0.001, gamma=0.9, tau=0.01):
        self.session = session
        self.n_actions = n_actions
        self.actors_eval_actions = actors_eval_actions
        self.actors_target_actions = actors_target_actions
        self.eval_states = eval_states
        self.target_states = target_states
        self.rewards = rewards

        with tf.variable_scope(scope):
            self.eval_values = self.build_network(self.eval_states,
                                                  self.actors_eval_actions,
                                                  'eval', trainable=True)
            self.target_values = self.build_network(self.target_states,
                                                    self.actors_target_actions,
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
        self.action_gradients = []
        for i in range(len(self.actors_eval_actions)):
            self.action_gradients.append(tf.gradients(ys=self.eval_values,
                                                      xs=self.actors_eval_actions[i])[0])

        self.update_target = [tf.assign(t, (1 - tau) * t + tau * e)
                              for t, e in zip(self.target_weights, self.eval_weights)]

    def build_network(self, x1, x2, scope, trainable):
        with tf.variable_scope(scope):
            W = tf.random_normal_initializer(0.0, 0.1)
            b = tf.constant_initializer(0.1)

            h3 = tf.zeros([x1[0].shape[0], 30])
            for i in range(len(x1)):
                h1 = tf.layers.dense(x1[i], 30, activation=tf.nn.relu,
                                     kernel_initializer=W, bias_initializer=b,
                                     name='h1-' + str(i), trainable=trainable)
                h21 = tf.get_variable('h21-' + str(i), [30, 30],
                                      initializer=W, trainable=trainable)
                h22 = tf.get_variable('h22-' + str(i), [self.n_actions[i], 30],
                                      initializer=W, trainable=trainable)

                h3 = h3 + tf.matmul(h1, h21) + tf.matmul(x2, h22)

            b2 = tf.get_variable('b2', [1, 30], initializer=b,
                                 trainable=trainable)
            h3 = tf.nn.relu(h3 + b2)
            values = tf.layers.dense(h3, 1, kernel_initializer=W,
                                     bias_initializer=b, name='values',
                                     trainable=trainable)

        return values

    def learn(self, states, actions, rewards, states_next):
        # TODO: all agents
        self.session.run(self.optimize, feed_dict={self.eval_states: states,
                                                   self.actors_eval_actions: actions,
                                                   self.rewards: rewards,
                                                   self.target_states: states_next})
        self.session.run(self.update_target)
