#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/26
# @Author  : Jiaxin Gao, Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================
import argparse
import ray
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


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
        self.x = []
        self.x.append([MLPNet(obs_dim, act_dim) for _ in range(T)])
        for i in range(N):
            the_first_time_step = initial_samples[i]
            i_th_sample_1 = [the_first_time_step] + [tf.Variable(initial_value=tf.ones(shape=(obs_dim,))) for _ in
                                                     range(T - 1)]
            i_th_sample_2 = [the_first_time_step] + [tf.Variable(initial_value=tf.ones(shape=(obs_dim,))) for _ in
                                                     range(T - 1)]
            i_th_sample = i_th_sample_1 + i_th_sample_2
            self.x.append(i_th_sample)

        # self.x: [[theta0, ..., theta_{T-1}],
        #          [x^0_{0, 1},..., x^0_{T-1, 1}, x^0_{0, 2},..., x^0_{T-1, 2}],
        #          [x^1_{0, 1},..., x^1_{T-1, 1}, x^1_{0, 2},..., x^1_{T-1, 2}],
        #          ...]


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
        self.all_parameter = ParameterContainer(args, initial_samples)
        self.T = args.T
        self.N = args.N
        self.rou = args.rou
        self.obs_dim = args.obs_dim
        self.act_dim = args.act_dim

    def update_all_para(self, new_para):
        self.all_parameter.assign_all(new_para)

    def dynamics(self, obs, action, dis):
        '''
        :param obs, shape(batch_size, obs_dim)
        :param action, shape(batch_size, act_dim)
        :return: next_obs, l
        '''
        sample_time = 0.02
        tau = 0.1
        A = [[0, 1, sample_time], [0, 0, 1], [0, 0, -1 / tau]]
        B = [[0, 0, 1 / tau]]
        C = tf.constant([[1, 0, 0]])
        y = tf.matmul(C, obs)

        next_obs = tf.add(tf.matmul(A, obs), tf.matmul(B, action))
        l = 1 / 2 * tf.square(y - dis)
        return next_obs, l

    def utility_func(self, obs, action):
        '''

        :param obs
        :return: l
        '''
        pass

    def construct_ith_loss(self, j):
        loss = 0
        if j == 0:
            theta_0 = self.all_parameter.x.x[0][0]
            all_x_02 = tf.reshape([self.all_parameter.x.x[i][self.T] for i in range(1, self.N+1)], shape=(self.N, self.obs_dim))
            all_y_11 = tf.reshape([self.all_parameter.y.x[i][1] for i in range(1, self.N+1)], shape=(self.N, self.obs_dim))
            all_x_11 = tf.reshape([self.all_parameter.x.x[i][1] for i in range(1, self.N+1)], shape=(self.N, self.obs_dim))
            all_z_1 = tf.reshape([self.all_parameter.z.z[i][1] for i in range(1, self.N+1)], shape=(self.N, self.obs_dim))
            y_theta_0 = self.all_parameter.y.x[0][0]
            z_theta = self.all_parameter.z.z[0]
            pi_x_02 = theta_0(all_x_02)
            dis = [1,2,4]
            x_11, ls = self.dynamics(all_x_02, pi_x_02, dis)
            loss += tf.reduce_mean(ls)
            loss += tf.reduce_sum(all_y_11*(all_z_1 - all_x_11))
            loss += tf.reduce_sum(y_theta_0*(z_theta - theta_0))
            print('loss', loss)
        else:
            theta = self.all_parameter.x.x[j][0]
            all_x_j2 = tf.reshape([self.all_parameter.x.x[i][self.T + j] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_y_j_add_1 = tf.reshape([self.all_parameter.y.x[i][j + 1] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_x_j_add_1 = tf.reshape([self.all_parameter.x.x[i][j + 1] for i in range(1, self.N + 1)],
                                  shape=(self.N, self.obs_dim))
            all_z_j = tf.reshape([self.all_parameter.z.z[i][j + 1] for i in range(1, self.N + 1)],
                                 shape=(self.N, self.obs_dim))
            y_theta_j = self.all_parameter.y.x[0][j]
            z_theta = self.all_parameter.z.z[0]
            pi_x_j2 = theta(all_x_j2)
            dis = [1, 2, 4]
            x_11, ls = self.dynamics(all_x_j2, pi_x_j2, dis)
            loss += tf.reduce_mean(ls)
            loss += tf.reduce_sum(all_y_j_add_1 * (all_z_j - all_x_j_add_1))
            loss += tf.reduce_sum(y_theta_j * (z_theta - theta))
            print('loss', loss)

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
        new_y = self.y + self.rou * (self.x - self.z)
        return new_y

    def update_z(self):
        new_z = []
        return new_z


def built_DADP_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='training') # training testing
    parser.add_argument('--T', type=int, default='2')
    parser.add_argument('--N', type=int, default='3')
    parser.add_argument('--rou', type=float, default='1')
    parser.add_argument('--obs_dim', type=int, default='3')
    parser.add_argument('--act_dim', type=int, default='1')
    parser.add_argument('max_iter', type=int, default='100')

    return parser.parse_args()


def main():
    args = built_DADP_parser()#T,N,rou,obs_dim,act_dim
    ray.init()
    #ray.init(object_store_memory=5120*1024*1024)
    #initial_samples = None
    initial_samples = [[1,2,3],[2,3,3],[5,6,7]]

    all_parameters = ParameterContainer(initial_samples, args)
    print('all_parameters.x=', all_parameters.x)
    exit()
    learners = Learner(initial_samples, args)
    learners.construct_ith_loss(0)
    #learners = [ray.remote(num_cpus=1)(Learner).remote(initial_samples, args) for _ in range(args.T)]
    exit()
    for _ in range(args.max_iter):
        # 1st step
        ray.get([learner.learn.remote() for learner in learners])

        # 2nd step
        # fetch the corresponding variables in each learner and concatenate them into x
        x = None
        all_parameters.assign_x(x)
        all_parameters.update_z()

        # 3rd
        all_parameters.update_y()

        # deal with this iteration
        # terminal judgement
        weights = ray.put(all_parameters.trainable_variables)
        for learner in learners:
            learner.update_all_para.remote(weights)


if __name__ == '__main__':
    main()
