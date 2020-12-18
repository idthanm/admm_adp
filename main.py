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
            i_th_sample_1 = [the_first_time_step] + [tf.Variable(initial_value=tf.ones(shape=(obs_dim,))) for _ in
                                                     range(T - 1)]
            i_th_sample_2 = [the_first_time_step] + [tf.Variable(initial_value=tf.ones(shape=(obs_dim,))) for _ in
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
            for_sample_i = [the_first_time_step] + [tf.Variable(initial_value=tf.ones(shape=(obs_dim,))) for _ in
                                                    range(T - 1)]
            self.z.append(for_sample_i)

        # self.z: [z_{theta},
        #          [z^0_0,...,z^0_{T-1}],
        #          [z^1_0,...,z^1_{T-1}],
        #          ...]


class Learner(object):
    def __init__(self, initial_samples, args):
        self.all_parameter = ParameterContainer(initial_samples, args)
        self.opt = tf.keras.optimizers.Adam(lr=3e-4)
        self.T = args.T
        self.N = args.N
        self.rou = args.rou
        self.tau = args.tau
        self.exp_v = args.exp_v
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim
        self.delay = args.delay
        self.iteration_number_x_update = args.iteration_number_x_update

    def update_all_para(self, new_para):
        self.all_parameter.assign_all(new_para)

    def dynamics(self, obs, action):
        '''
        :param obs, shape(batch_size, obs_dim)
        :param action, shape(batch_size, act_dim)
        :return: next_obs, l
        '''
        vs, a_xs = obs[:, 0], obs[:, 1]
        us = action[:, 0]
        deri = [a_xs, 1. / (self.delay*(us-a_xs))]
        deri = tf.reshape(deri, shape=(self.obs_dim, self.N))
        deri = tf.transpose(deri)
        next_obs = obs + self.delay * deri
        print('next_obs=', next_obs)
        l = 0.5*tf.reduce_mean(tf.square(vs-self.exp_v)) + 0.001*tf.reduce_mean(tf.square(us))
        return next_obs, l

    def construct_ith_loss(self, j):
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
            loss += tf.reduce_mean(ls)
            loss += tf.reduce_sum(all_y_11 * (all_z_1 - x_11))
            loss += tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                                   for theta_in_y, theta_in_z, theta_in_x in
                                   zip(y_mlp_0.trainable_weights, z_mlp.trainable_weights, x_mlp_0.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(all_z_1 - x_11))
            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z-theta_in_x))
                                   for theta_in_z, theta_in_x in
                                   zip(z_mlp.trainable_weights, x_mlp_0.trainable_weights)])
            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][1].assign(x_11[i - 1, :])
            all_x_11 = tf.reshape([self.all_parameter.x.params[i][1] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_x_02 = tf.reshape([self.all_parameter.x.params[i][self.T] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            #print('all_x_11=', all_x_11)
            #print('loss=', loss)
            return loss, x_mlp_0.variables, all_x_11, all_x_02
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
            loss += tf.reduce_sum(all_y_T_1_2 * (all_z_T_1 - all_x_T_1_2))
            loss += tf.reduce_sum([tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                                   for theta_in_y, theta_in_z, theta_in_x in
                                   zip(y_mlp_T_1.trainable_weights, z_mlp.trainable_weights, x_mlp_T_1.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(all_z_T_1 - all_x_T_1_2))
            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z-theta_in_x))
                                   for theta_in_z, theta_in_x in
                                   zip(z_mlp.trainable_weights, x_mlp_T_1.trainable_weights)])
            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][2*self.T - 1].assign(all_x_T_1_2[i - 1, :])

            all_x_T_1 = tf.reshape([self.all_parameter.x.params[i][self.T-1] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_x_T_1_2 = tf.reshape([self.all_parameter.x.params[i][2*self.T - 1] for i in range(1, self.N + 1)],
                                   shape=(self.N, self.obs_dim))
            return loss, x_mlp_T_1.variables, all_x_T_1, all_x_T_1_2
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
            loss += tf.reduce_sum(all_y_j_add_1 * (all_z_j_add_1 - x_j_add_1))
            loss += tf.reduce_sum(all_y_j2 * (all_z_j - all_x_j2))
            loss += tf.reduce_sum(
                [tf.reduce_sum(tf.stop_gradient(theta_in_y) * (tf.stop_gradient(theta_in_z) - theta_in_x))
                 for theta_in_y, theta_in_z, theta_in_x in
                 zip(y_mlp_j.trainable_weights, z_mlp.trainable_weights, x_mlp_j.trainable_weights)])
            loss += self.rou / 2 * tf.reduce_sum(tf.square(all_z_j_add_1 - x_j_add_1))
            loss += self.rou / 2 * tf.reduce_sum(tf.square(all_z_j - all_x_j2))
            loss += self.rou / 2 * tf.reduce_sum([tf.reduce_sum(tf.square(theta_in_z - theta_in_x))
                                                  for theta_in_z, theta_in_x in
                                                  zip(z_mlp.trainable_weights, x_mlp_j.trainable_weights)])
            for i in range(1, self.N + 1):
                self.all_parameter.x.params[i][1].assign(x_j_add_1[i - 1, :])
            all_x_j_add_1 = tf.reshape([self.all_parameter.x.params[i][j + 1] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))
            all_x_j2 = tf.reshape([self.all_parameter.x.params[i][self.T + j] for i in range(1, self.N + 1)],
                                       shape=(self.N, self.obs_dim))
            return loss, x_mlp_j.variables, all_x_j_add_1, all_x_j2

    def assign_x(self, new_xs):
        for new_x, local_x in zip(new_xs, self.x.trainable_variables):
            local_x.assign(new_x)

    def assign_y(self, new_ys):
        for new_y, local_y in zip(new_ys, self.y.trainable_variables):
            local_y.assign(new_y)

    def assign_z(self, new_zs):
        for new_z, local_z in zip(new_zs, self.z.trainable_variables):
            local_z.assign(new_z)

    def learn(self, j):
        for _ in range(self.iteration_number_x_update):
            with tf.GradientTape() as tape:
                loss, x_mlp_value, updated_x_j_add_1, updated_x_j2 = self.construct_ith_loss(j)
            print('loss=', loss)
            #exit()
            grad = tape.gradient(loss, self.all_parameter.trainable_variables)
            self.opt.apply_gradients(grads_and_vars=zip(grad, self.all_parameter.trainable_variables))
        return x_mlp_value, updated_x_j_add_1, updated_x_j2

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
        self.z.z[0].set_weights(mean_z)
        for i in range(1, self.N + 1):
            for j in range(1, self.T):
                self.z.z[i][j].assign(0.5*(self.x.params[i][j] + self.x.params[i][j + self.T]))


def built_DADP_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    parser.add_argument('--T', type=int, default='3')
    parser.add_argument('--N', type=int, default='3')
    parser.add_argument('--rou', type=float, default='1')
    parser.add_argument('--obs_dim', type=int, default='2')
    parser.add_argument('--act_dim', type=int, default='1')
    parser.add_argument('--max_iter', type=int, default='100')
    parser.add_argument('--tau', type=float, default='0.02') # sample_time
    parser.add_argument('--delay', type=float,default='0.35') # 执行机构延迟时间
    parser.add_argument('--exp_v', type=float, default='3.0')
    parser.add_argument('--iteration_number_x_update', type=int, default='30')
    return parser.parse_args()


def main():
    args = built_DADP_parser() #params_init
    initial_samples = [[1., 2.], [2., 3.], [3., 4.]]
    all_parameters = ParameterContainer(initial_samples, args)
    learners = Learner(initial_samples, args)
    # 由子进程中的 x 更新主进程中的 x
    for _ in range(args.max_iter):
        # 1st step update x
        for time_horizon in range(args.T):
            x_mlp_value, updated_x_j_add_1, updated_x_j2 = learners.learn(time_horizon)
            all_parameters.x.params[0][time_horizon] = x_mlp_value
            if time_horizon == 0:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][1].assign(updated_x_j_add_1[i - 1, :])
            elif time_horizon == args.T - 1:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][2 * args.T - 1].assign(updated_x_j2[i - 1, :])
            else:
                for i in range(1, args.N + 1):
                    all_parameters.x.params[i][time_horizon + 1].assign(updated_x_j_add_1[i - 1, :])
                    all_parameters.x.params[i][args.T + 1].assign(updated_x_j2[i - 1, :])
        #print('x=', all_parameters.x.trainable_variables)
        #exit()
        # 2nd step update z

        #all_parameters.assign_x(x)
        all_parameters.update_z()

        # 3rd
        all_parameters.update_y()
        #convergence condition unfinished
        convergence = 0
        if convergence == 1:
            break
        else:
            pass

        learners.all_parameter.assign_x(all_parameters.x.trainable_variables)

        learners.all_parameter.assign_y(all_parameters.y.trainable_variables)

        learners.all_parameter.assign_z(all_parameters.z.trainable_variables)

        # deal with this iteration
        # terminal judgement
        #weights = ray.put(all_parameters.trainable_variables)
        #weights = all_parameters.trainable_variables
        #for learner in learners:
        # learner.update_all_para.remote(weights)


if __name__ == '__main__':
    main()
