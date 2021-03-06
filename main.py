#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/26
# @Author  : Jiaxin Gao, Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================
import argparse
#import ray
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import numpy as np
import datetime

class MLPNet(Model):
    def __init__(self, input_dim, output_dim):
        super(MLPNet, self).__init__()
        self.first_ = Dense(output_dim,
                            activation='relu',
                            kernel_initializer=tf.keras.initializers.Orthogonal(1.),
                            bias_initializer=tf.keras.initializers.Constant(0.),
                            dtype=tf.float32)

        self.build(input_shape=(None, input_dim))

    def call(self, x, **kwargs):
        x = self.first_(x)
        return x


class x_or_y_module(tf.Module):
    def __init__(self, args, initial_samples):
        super().__init__()
        T = args.T
        N = args.N
        obs_dim = args.obs_dim
        act_dim = args.act_dim
        self.params = []
        self.params.append([MLPNet(obs_dim, act_dim) for _ in range(T)])
        for i in range(N):
            the_first_time_step = initial_samples[i]
            i_th_sample_1 = [the_first_time_step] + [tf.Variable(initial_value=tf.zeros(shape=(obs_dim,))) for _ in
                                                     range(T - 1)]
            i_th_sample_2 = [the_first_time_step] + [tf.Variable(initial_value=tf.zeros(shape=(obs_dim,))) for _ in
                                                     range(T - 1)]
            i_th_sample = i_th_sample_1 + i_th_sample_2
            self.params.append(i_th_sample)

        # self.params: [[theta0, ..., theta_{T-1}],
        #              [x^0_{0, 1},..., x^0_{T-1, 1}, x^0_{0, 2},..., x^0_{T-1, 2}],
        #              [x^1_{0, 1},..., x^1_{T-1, 1}, x^1_{0, 2},..., x^1_{T-1, 2}],
        #              ...]


class z_module(tf.Module):
    def __init__(self, args, initial_samples):
        super().__init__()
        T = args.T
        N = args.N
        obs_dim = args.obs_dim
        act_dim = args.act_dim
        self.z = []
        self.z.append(MLPNet(obs_dim, act_dim))
        for i in range(N):
            the_first_time_step = initial_samples[i]
            #print('the_first_time_step = ', the_first_time_step)
            for_sample_i = [the_first_time_step] + [tf.Variable(initial_value=tf.zeros(shape=(obs_dim,))) for _ in
                                                    range(T - 1)]
            #print('for_sample_i = ', for_sample_i)
            #exit()
            self.z.append(for_sample_i)

        # self.z: [z_{theta},
        #          [z^0_0,...,z^0_{T-1}],
        #          [z^1_0,...,z^1_{T-1}],
        #          ...]


class Learner(object):
    def __init__(self, initial_samples, args):
        self.all_parameter = ParameterContainer(initial_samples, args)
        self.opt = tf.keras.optimizers.Adam(args.lr)
        self.T = args.T
        self.N = args.N
        self.rou = args.rou
        self.tau = args.tau
        self.exp_v = args.exp_v
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.delay = args.delay
        self.iteration_number_x_update = args.iteration_number_x_update
        self.eps_abs = args.eps_abs
        self.eps_rel = args.eps_rel

    def update_all_para(self, new_para):
        self.all_parameter.assign_all(new_para)

    def dynamics(self, obs, action):
        vs = obs[:, 0]
        us = action[:, 0]
        #('vs = ', vs)
        #print('us = ', us)

        deri = [us]
        deri = tf.reshape(deri, shape=(self.obs_dim, self.N))
        deri = tf.transpose(deri)
        print('deri = ', deri)
        next_obs = obs + self.tau * deri
        #print('next_obs = ', next_obs)
        #exit()
        #print('next_obs=', next_obs)
        l = 0.5*tf.reduce_mean(tf.square(next_obs - self.exp_v)) #+ 0.5*tf.reduce_mean(tf.square(us))
        print('0.5*tf.reduce_mean(tf.square(next_obs - self.exp_v)) = ', 0.5*tf.reduce_mean(tf.square(next_obs - self.exp_v)))
        #print('0.005*tf.reduce_mean(tf.square(us)) = ', 0.5*tf.reduce_mean(tf.square(us)))

        return next_obs, l

    def construct_ith_loss(self, j, summary_writer, idx):
        loss = 0
        if j == 0:
            x_mlp_0 = self.all_parameter.x.params[0][0]
            all_x_02 = tf.reshape([self.all_parameter.x.params[i][self.T] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_y_11 = tf.reshape([self.all_parameter.y.params[i][1] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_z_1 = tf.reshape([self.all_parameter.z.z[i][1] for i in range(1, self.N + 1)],
                                 shape=(self.N, self.obs_dim))
            y_mlp_0 = self.all_parameter.y.params[0][0]
            z_mlp = self.all_parameter.z.z[0]
            pi_x_02 = x_mlp_0(all_x_02)
            x_11, ls = self.dynamics(all_x_02, pi_x_02)

            wight = np.array(np.array(x_mlp_0.get_weights()))

            with summary_writer.as_default():
                tf.summary.scalar('weight = ', wight[0][0], idx, None)
                tf.summary.scalar('bias = ', wight[1][0], idx, None)
                tf.summary.scalar('pi_x_02', pi_x_02.numpy()[0][0], idx, None)
                #tf.summary.scalar('all_x_02', all_x_02.numpy()[0][0], idx, None)
                tf.summary.scalar('x_11', x_11.numpy()[0][0], idx, None)
                tf.summary.scalar('ls', ls, idx, None)
            loss += tf.reduce_mean(ls)

            loss += tf.reduce_sum(tf.stop_gradient(all_y_11) * (tf.stop_gradient(all_z_1) - x_11))

            loss += tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                                   for theta_in_y, theta_in_z, theta_in_x in
                                   zip(y_mlp_0.trainable_weights, z_mlp.trainable_weights, x_mlp_0.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(tf.stop_gradient(all_z_1) - x_11))

            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(tf.stop_gradient(theta_in_z)-theta_in_x))
                                   for theta_in_z, theta_in_x in
                                   zip(z_mlp.trainable_weights, x_mlp_0.trainable_weights)])
            with summary_writer.as_default():
                tf.summary.scalar('y(z-x)', tf.reduce_sum(tf.stop_gradient(all_y_11) * (tf.stop_gradient(all_z_1) - x_11)), idx, None)
                tf.summary.scalar('y(z-x)_theta', tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))for theta_in_y, theta_in_z, theta_in_x in zip(y_mlp_0.trainable_weights, z_mlp.trainable_weights, x_mlp_0.trainable_weights)]), idx, None)
                tf.summary.scalar('(z-x)^2', self.rou / 2 * tf.reduce_sum(tf.square(tf.stop_gradient(all_z_1) - x_11)), idx, None)
                tf.summary.scalar('(z-x)^2_theta', self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(tf.stop_gradient(theta_in_z) - theta_in_x)) for theta_in_z, theta_in_x in zip(z_mlp.trainable_weights, x_mlp_0.trainable_weights)]), idx, None)

            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][1].assign(x_11[i - 1, :])
            return loss
        elif j == self.T - 1:
            x_mlp_T_1 = self.all_parameter.x.params[0][self.T - 1]
            all_x_T_1_2 = tf.reshape([self.all_parameter.x.params[i][2 * self.T - 1] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
            all_y_T_1_2 = tf.reshape([self.all_parameter.y.params[i][2 * self.T - 1] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
            all_z_T_1 = tf.reshape([self.all_parameter.z.z[i][self.T - 1] for i in range(1, self.N + 1)],
                                   shape=(self.N, self.obs_dim))
            y_mlp_T_1 = self.all_parameter.y.params[0][self.T - 1]
            z_mlp = self.all_parameter.z.z[0]
            pi_x_T_1_2 = x_mlp_T_1(all_x_T_1_2)
            x_T_1, ls = self.dynamics(all_x_T_1_2, pi_x_T_1_2)
            loss += tf.reduce_mean(ls)
            loss += tf.reduce_sum(tf.stop_gradient(all_y_T_1_2) * (tf.stop_gradient(all_z_T_1) - all_x_T_1_2))
            loss += tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                                   for theta_in_y, theta_in_z, theta_in_x in
                                   zip(y_mlp_T_1.trainable_weights, z_mlp.trainable_weights, x_mlp_T_1.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(tf.stop_gradient(all_z_T_1) - all_x_T_1_2))
            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(tf.stop_gradient(theta_in_z) - theta_in_x))
                                   for theta_in_z, theta_in_x in
                                   zip(z_mlp.trainable_weights, x_mlp_T_1.trainable_weights)])
            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][2*self.T - 1].assign(all_x_T_1_2[i - 1, :])
            return loss
        else:
            x_mlp_j = self.all_parameter.x.params[0][j]
            all_x_j2 = tf.reshape([self.all_parameter.x.params[i][self.T + j] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_y_j_add_1 = tf.reshape([self.all_parameter.y.params[i][j + 1] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))
            all_z_j_add_1 = tf.reshape([self.all_parameter.z.z[i][j + 1] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))
            all_y_j2 = tf.reshape([self.all_parameter.y.params[i][self.T + j] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_z_j = tf.reshape([self.all_parameter.z.z[i][j] for i in range(1, self.N + 1)],
                                 shape=(self.N, self.obs_dim))
            y_mlp_j = self.all_parameter.y.params[0][j]
            z_mlp = self.all_parameter.z.z[0]
            pi_x_j2 = x_mlp_j(all_x_j2)
            x_j_add_1, ls = self.dynamics(all_x_j2, pi_x_j2,)
            loss += tf.reduce_mean(ls)
            loss += tf.reduce_sum(tf.stop_gradient(all_y_j_add_1) * (tf.stop_gradient(all_z_j_add_1) - x_j_add_1))
            loss += tf.reduce_sum(tf.stop_gradient(all_y_j2) * (tf.stop_gradient(all_z_j) - all_x_j2))
            loss += tf.reduce_sum(
                [tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                 for theta_in_y, theta_in_z, theta_in_x in
                 zip(y_mlp_j.trainable_weights, z_mlp.trainable_weights, x_mlp_j.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(tf.stop_gradient(all_z_j_add_1) - x_j_add_1))
            loss += self.rou / 2 * tf.reduce_sum(tf.square(tf.stop_gradient(all_z_j) - all_x_j2))
            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(tf.stop_gradient(theta_in_z) - theta_in_x))
                                                  for theta_in_z, theta_in_x in
                                                  zip(z_mlp.trainable_weights, x_mlp_j.trainable_weights)])
            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][1].assign(x_j_add_1[i - 1, :])
            return loss

    def assign_x(self, new_xs):
        for new_x, local_x in zip(new_xs, self.x.trainable_variables):
            local_x.assign(new_x)

    def assign_y(self, new_ys):
        for new_y, local_y in zip(new_ys, self.y.trainable_variables):
            local_y.assign(new_y)

    def assign_z(self, new_zs):
        for new_z, local_z in zip(new_zs, self.z.trainable_variables):
            local_z.assign(new_z)

    def learn(self, j, summary_writer):
        for idx in range(self.iteration_number_x_update):
            with tf.GradientTape() as tape:
                loss = self.construct_ith_loss(j, summary_writer, idx)

            if (idx % 10 == 0):
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, idx, None)

            #print('loss=', loss)
            #exit()
            grad = tape.gradient(loss, self.all_parameter.trainable_variables)
            self.opt.apply_gradients(grads_and_vars=zip(grad, self.all_parameter.trainable_variables))

        x_mlp_value = self.all_parameter.x.params[0][j].trainable_variables

        updated_x_j_add_1 = tf.reshape([self.all_parameter.x.params[i][j + 1] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))

        updated_x_j2 = tf.reshape([self.all_parameter.x.params[i][self.T + j] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))

        return x_mlp_value, updated_x_j_add_1, updated_x_j2

    def terminal(self, old_z):
        ss = 0
        for j in range(self.T):
            z_mlp = old_z.z[0]
            z_mlp_updated = self.all_parameter.z.z[0]
            ss += self.rou * tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_updated_z - theta_in_z))
                                           for theta_in_updated_z, theta_in_z in
                                           zip(z_mlp_updated.trainable_variables, z_mlp.trainable_variables)])

            all_z_j_updated = tf.reshape([self.all_parameter.z.z[i][j] for i in range(1, self.N + 1)],
                                 shape=(self.N, self.obs_dim))

            all_z_j = tf.reshape([old_z.z[i][j] for i in range(1, self.N + 1)],
                                 shape=(self.N, self.obs_dim))
            ss += self.rou * tf.reduce_sum(tf.square(all_z_j_updated - all_z_j))
        ss = 0.5 * ss
        AX = 0
        BZ = 0
        r = 0
        for j in range(self.T):
            if j == 0:
                x_mlp_0 = self.all_parameter.x.params[0][0]
                all_z_1 = tf.reshape([self.all_parameter.z.z[i][1] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
                all_x_11 = tf.reshape([self.all_parameter.x.params[i][1] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
                z_mlp = self.all_parameter.z.z[0]
                AX += tf.reduce_sum(tf.square(all_x_11))
                BZ += tf.reduce_sum(tf.square(all_z_1))
                r += tf.reduce_sum(tf.square(all_x_11 - all_z_1))
                AX += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_x))
                                                      for theta_in_x in
                                                      zip(x_mlp_0.trainable_weights)])
                BZ += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z))
                                                      for theta_in_z in
                                                      zip(z_mlp.trainable_weights)])
                r += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z - theta_in_x))
                                                      for theta_in_z, theta_in_x in
                                                      zip(z_mlp.trainable_weights, x_mlp_0.trainable_weights)])
            elif j == self.T - 1:
                x_mlp_T_1 = self.all_parameter.x.params[0][self.T - 1]
                all_x_T_1_2 = tf.reshape([self.all_parameter.x.params[i][2 * self.T - 1] for i in range(1, self.N + 1)],
                                         shape=(self.N, self.obs_dim))
                all_z_T_1 = tf.reshape([self.all_parameter.z.z[i][self.T - 1] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))
                z_mlp = self.all_parameter.z.z[0]

                AX += tf.reduce_sum(tf.square(all_x_T_1_2))
                BZ += tf.reduce_sum(tf.square(all_z_T_1))
                r += tf.reduce_sum(tf.square(all_z_T_1 - all_x_T_1_2))
                AX += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_x))
                                     for theta_in_x in
                                     zip(x_mlp_T_1.trainable_weights)])
                BZ += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z))
                                     for theta_in_z in
                                     zip(z_mlp.trainable_weights)])
                r += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z - theta_in_x))
                                    for theta_in_z, theta_in_x in
                                    zip(z_mlp.trainable_weights, x_mlp_T_1.trainable_weights)])
            else:
                x_mlp_j = self.all_parameter.x.params[0][j]
                all_x_j2 = tf.reshape([self.all_parameter.x.params[i][self.T + j] for i in range(1, self.N + 1)],
                                      shape=(self.N, self.obs_dim))
                all_z_j_add_1 = tf.reshape([self.all_parameter.z.z[i][j + 1] for i in range(1, self.N + 1)],
                                           shape=(self.N, self.obs_dim))
                all_z_j = tf.reshape([self.all_parameter.z.z[i][j] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
                z_mlp = self.all_parameter.z.z[0]
                all_x_j_add_1 = tf.reshape([self.all_parameter.x.params[i][j + 1] for i in range(1, self.N + 1)],
                                      shape=(self.N, self.obs_dim))

                AX += tf.reduce_sum(tf.square(all_x_j_add_1))
                BZ += tf.reduce_sum(tf.square(all_z_j_add_1))
                r += tf.reduce_sum(tf.square(all_z_j_add_1 - all_x_j_add_1))
                AX += tf.reduce_sum(tf.square(all_x_j2))
                BZ += tf.reduce_sum(tf.square(all_z_j))
                r += tf.reduce_sum(tf.square(all_z_j - all_x_j2))
                AX += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_x))
                                     for theta_in_x in
                                     zip(x_mlp_j.trainable_weights)])
                BZ += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z))
                                     for theta_in_z in
                                     zip(z_mlp.trainable_weights)])
                r += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z - theta_in_x))
                                    for theta_in_z, theta_in_x in
                                    zip(z_mlp.trainable_weights, x_mlp_j.trainable_weights)])

        r **= 0.5
        AX **= 0.5
        BZ **= 0.5
        AL = 0
        for k in range(0, self.T):
            if j == 0:
                y_mlp_0 = self.all_parameter.y.params[0][0]
                all_y_11 = tf.reshape([self.all_parameter.y.params[i][1] for i in range(1, self.N + 1)],
                                     shape=(self.N, self.obs_dim))
                AL += tf.reduce_sum(tf.square(all_y_11))
                AL += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_y))
                                                      for theta_in_y in
                                                      zip(y_mlp_0.trainable_weights)])
            elif j == self.T - 1:
                y_mlp_T_1 = self.all_parameter.x.params[0][self.T - 1]
                all_y_T_1_2 = tf.reshape([self.all_parameter.x.params[i][2 * self.T - 1] for i in range(1, self.N + 1)],
                                         shape=(self.N, self.obs_dim))
                AL += tf.reduce_sum(tf.square(all_y_T_1_2))
                AL += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_y))
                                     for theta_in_y in
                                     zip(y_mlp_T_1.trainable_weights)])
            else:
                y_mlp_j = self.all_parameter.x.params[0][j]
                all_y_j2 = tf.reshape([self.all_parameter.x.params[i][self.T + j] for i in range(1, self.N + 1)],
                                      shape=(self.N, self.obs_dim))
                all_y_j_add_1 = tf.reshape([self.all_parameter.x.params[i][j + 1] for i in range(1, self.N + 1)],
                                      shape=(self.N, self.obs_dim))

                AL += tf.reduce_sum(tf.square(all_y_j_add_1))
                AL += tf.reduce_sum(tf.square(all_y_j2))
                AL += tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_y))
                                     for theta_in_y in
                                     zip(y_mlp_j.trainable_weights)])
        AL **= 0.5
        eps_pri = np.sqrt(9 * self.T) * self.eps_abs + self.eps_rel * max(AX, BZ)
        eps_dual = np.sqrt(9 * self.T) * self.eps_abs + self.eps_rel * AL
        return ss, eps_pri, eps_dual, r


class ParameterContainer(tf.Module):
    def __init__(self, initial_samples, args):
        super().__init__()
        self.T = args.T
        self.N = args.N
        self.rou = args.rou
        self.x = x_or_y_module(args, initial_samples)
        self.z = z_module(args, initial_samples)
        self.y = x_or_y_module(args, initial_samples)

    def assign_all(self, new_variables):
        for new_var, local_var in zip(new_variables, self.trainable_variables):   #assign() 函数，用新值替换旧值。
            local_var.assign(new_var)

    def assign_x(self, new_xs):
        for new_x, local_x in zip(new_xs, self.x.trainable_variables):
            local_x.assign(new_x)

    def assign_y(self, new_ys):
        for new_y, local_y in zip(new_ys, self.y.trainable_variables):
            local_y.assign(new_y)

    def assign_z(self, new_zs):
        for new_z, local_z in zip(new_zs, self.z.trainable_variables):
            local_z.assign(new_z)

    def update_y(self):
        for j in range(0, self.T):
            for theta_in_y, theta_in_x, theta_in_z in zip(self.y.params[0][j].trainable_variables, self.x.params[0][j].trainable_variables, self.z.z[0].trainable_variables):
                theta_in_y.assign(theta_in_y + self.rou * (theta_in_x-theta_in_z))
        for i in range(1, self.N + 1):
            for j in range(1, self.T):
                self.y.params[i][j].assign(self.y.params[i][j] + self.rou * (self.x.params[i][j] - self.z.z[i][j]))
            for j in range(self.T+1, 2 * self.T):
                self.y.params[i][j].assign(self.y.params[i][j] + self.rou * (self.x.params[i][j] - self.z.z[i][j - self.T]))

    def update_z(self):
        all_zs = np.array([np.array(self.x.params[0][i].get_weights()) for i in range(0, self.T)])
        mean_z = np.mean(all_zs, axis=0)
        weights = [mean_z[0].reshape(1, 1), mean_z[1].reshape(1,)]
        self.z.z[0].set_weights(weights)
        #exit()
        for i in range(1, self.N + 1):
            for j in range(1, self.T):
                self.z.z[i][j].assign(0.5*(self.x.params[i][j] + self.x.params[i][j + self.T]))


def built_DADP_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='training') # training testing
    parser.add_argument('--T', type=int, default=15)
    parser.add_argument('--N', type=int, default=1)
    parser.add_argument('--rou', type=float, default=1)
    parser.add_argument('--obs_dim', type=int, default=1)
    parser.add_argument('--act_dim', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=30)
    parser.add_argument('--tau', type=float, default=0.1) # sample_time
    parser.add_argument('--delay', type=float,default=0.35) # 执行机构延迟时间
    parser.add_argument('--exp_v', type=float, default=1.0)
    parser.add_argument('--iteration_number_x_update', type=int, default=1000)
    parser.add_argument('--eps_abs', type=float, default=0.001)
    parser.add_argument('--eps_rel', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=3e-3)
    return parser.parse_args()


def main():
    args = built_DADP_parser() #params_init
    initial_samples = [[0.]]
    all_parameters = ParameterContainer(initial_samples, args) #main process
    learners = Learner(initial_samples, args)
    log_dir = "/home/chb/jupyter/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_writer = tf.summary.create_file_writer(log_dir)
    # 由子进程中的 x 更新主进程中的 x
    for idx in range(args.max_iter):
        # 1st step update x
        for time_horizon in range(args.T):
            x_mlp_value, updated_x_j_add_1, updated_x_j2 = learners.learn(time_horizon, summary_writer)
            print('time_horizon=', time_horizon)
            print('updated_x_j_add_1=', updated_x_j_add_1)
            print('updated_x_j2=', updated_x_j2)
            exit()
            for v1, v2 in zip(all_parameters.x.params[0][time_horizon].trainable_variables, x_mlp_value):
                v1.assign(v2)
            if time_horizon == 0:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][1].assign(updated_x_j_add_1[i - 1, :])
            elif time_horizon == args.T - 1:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][2 * args.T - 1].assign(updated_x_j2[i - 1, :])
            else:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][time_horizon + 1].assign(updated_x_j_add_1[i - 1, :])
                    all_parameters.x.params[i][args.T + time_horizon].assign(updated_x_j2[i - 1, :])
        # 2nd step update z
        #exit()
        all_parameters.update_z()
        # 3rd
        all_parameters.update_y()

        before_update_z = learners.all_parameter.z

        # main process to child process
        learners.all_parameter.assign_x(all_parameters.x.trainable_variables)
        learners.all_parameter.assign_y(all_parameters.y.trainable_variables)
        learners.all_parameter.assign_z(all_parameters.z.trainable_variables)

        # terminal judgement

        ss, eps_pri, eps_dual, r = learners.terminal(before_update_z)
        '''
        print('ss=', ss)
        print('eps_pri=', eps_pri)
        print('eps_dual=', eps_dual)
        print('r=', r)
        with summary_writer.as_default():
            tf.summary.scalar('ss = ', ss, idx, None)
            tf.summary.scalar('eps_pri = ', eps_pri, idx, None)
            tf.summary.scalar('eps_dual = ', eps_dual, idx, None)
            tf.summary.scalar('r = ', r, idx, None)
        '''
        if r <= eps_pri and ss <= eps_dual:
            print('success')
            break


if __name__ == '__main__':
    main()
