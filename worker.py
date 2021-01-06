#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================
# @Time    : 2020/11/26
# @Author  : Jiaxin Gao, Yang Guan (Tsinghua Univ.)
# @FileName: worker.py
# =====================================

#多层感知机和CNN
'''
import numpy as np
import tensorflow as tf

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1] #axis=-1指在最后一维增加维度
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]  astype() 数据转换类型函数
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0] # shape() 查看维数

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)  # randint(a,b) 生产a，b之间的一个随机数
        #print('index=', index)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


num_epochs = 5
batch_size = 50
learning_rate = 0.001

model = CNN()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

num_batches = int(data_loader.num_train_data)
#print('num_batches=', num_batches)
#print('range(num_batches)=', range(num_batches))

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
 #   exit()
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
'''


#Deep Q-learning
'''
import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

num_episodes = 500
num_exploration_episodes = 100
max_len_episode = 1000
batch_size = 32
learning_rate = 1e-3
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01


# Q-network用于拟合Q函数，和前节的多层感知机类似。输入state，输出各个action下的Q-value（CartPole下为2维）。
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(units=2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def predict(self, inputs):
        q_values = self.call(inputs)
        return tf.argmax(q_values, axis=-1)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')  # 实例化一个游戏环境，参数为游戏名称
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    replay_buffer = deque(maxlen=10000) # deque()是一个容器，可以从两端pop()或者append()。maxlen()定义了容器data数量的上界。
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state = env.reset()  # 初始化环境，获得初始状态
        epsilon = max(initial_epsilon * (num_exploration_episodes - episode_id) / num_exploration_episodes, final_epsilon)
        for t in range(max_len_episode):
            env.render()  # 对当前帧进行渲染，绘图到屏幕
            if random.random() < epsilon:  # epsilon-greedy探索策略
                action = env.action_space.sample()  # 以epsilon的概率选择随机动作
            else:
                #print('state=', state)
                action = model.predict(tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
                #print('action=', action)
                action = action[0]
                #print('action0=', action)
            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            next_state, reward, done, info = env.step(action)
            #print('next_state=',next_state)
            #print('reward=', reward)
            #print('done=', done)
            # 如果游戏Game Over，给予大的负奖励
            reward = -10. if done else reward
            # 将(state, action, reward, next_state)的四元组（外加done标签表示是否结束）放入经验重放池
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # 更新当前state
            state = next_state

            if done:  # 游戏结束则退出本轮循环，进行下一个episode
                print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                # 从经验回放池中随机取一个批次的四元组，并分别转换为NumPy数组
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *random.sample(replay_buffer, batch_size))
                batch_state, batch_reward, batch_next_state, batch_done = \
                    [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)

                q_value = model(tf.constant(batch_next_state, dtype=tf.float32))

                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 按照论文计算y值
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(  # 最小化y和Q-value的距离
                        y_true=y,
                        y_pred=tf.reduce_sum(model(tf.constant(batch_state)) * tf.one_hot(batch_action, depth=2),
                                             axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))  # 计算梯度并更新参数
'''


# para save and reload test
'''
import tensorflow as tf
import numpy as np
import argparse


class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1] #axis=-1指在最后一维增加维度
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]  astype() 数据转换类型函数
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0] # shape() 查看维数

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)  # randint(a,b) 生产a，b之间的一个随机数
        #print('index=', index)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--mode', default='test', help='train or test')
parser.add_argument('--num_epochs', default=1)
parser.add_argument('--batch_size', default=50)
parser.add_argument('--learning_rate', default=0.001)

args = parser.parse_args()
data_loader = MNISTLoader()


def train():
    model = MLP()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    num_batches = int(data_loader.num_train_data)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model)
    # 使用tf.train.CheckpointManager管理Checkpoint
    manager = tf.train.CheckpointManager(checkpoint, directory='./save', max_to_keep=3)
    for batch_index in range(num_batches):
        X, y = data_loader.get_batch(args.batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch %d: loss %f" % (batch_index, loss.numpy()))
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        if batch_index % 100 == 0:
            # 使用CheckpointManager保存模型参数到文件并自定义编号
            path = manager.save(checkpoint_number=batch_index)
            print("model saved to %s" % path)


def test():
    model_to_be_restored = MLP()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model_to_be_restored)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))
    y_pred = np.argmax(model_to_be_restored.predict(data_loader.test_data), axis=-1)
    print("test accuracy: %f" % (sum(y_pred == data_loader.test_label) / data_loader.num_test_data))


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    if args.mode == 'test':
        test()
'''

#可视化测试
'''
import tensorflow as tf
import numpy as np

class MNISTLoader():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1] #axis=-1指在最后一维增加维度
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]  astype() 数据转换类型函数
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0] # shape() 查看维数

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)  # randint(a,b) 生产a，b之间的一个随机数
        #print('index=', index)
        return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output


num_batches = 60000
batch_size = 50
learning_rate = 0.001
model = MLP()
data_loader = MNISTLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

summary_writer = tf.summary.create_file_writer('./tensorboard')  # 实例化记录器

for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
        with summary_writer.as_default():  # 指定记录器
            tf.summary.scalar("loss", loss, step=batch_index)  # 将当前损失函数的值写入记录器
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
'''


'''
#compare 1 and 2
for i in range(1):
    B = A[i, :]
    print('B=', B)
    for p in range(5):
        print('p=', p)
        compare = [1, 2, 3, 4, 5]
        opt1 = compare[p]
        compare.pop(p)
        print('opt1=', opt1)
        print('compare=', compare)
        for j in range(4):
            opt2 = compare[j]
            print('opt2=', opt2)
            count = 0
            for q in range(5):
                if B[q] == opt1:
                    count += 1
                    break
                elif B[q] == opt2:
                    count += 0
                    break
            print('count=', count)
            if count >= 5:
                score[0, p] += 1
                print('score[0,p]=', score[0,p])
'''
#compare 1 and 2

# 投票
'''
import numpy as np
import argparse

def vote():
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default='9')
    #parser.add_argument('--opt1', type=int, default='1')
    #parser.add_argument('--opt2', type=int, default='2')
    return parser.parse_args()

def para_ini():
    A = np.zeros((args.N, 5))
    A[0, :] = [3, 1, 4, 2, 5]  # xiaoli
    A[1, :] = [3, 4, 1, 2, 5]  # chaoyang
    A[2, :] = [3, 2, 5, 1, 4]  # zihao
    A[3, :] = [2, 4, 3, 1, 5]  # junheng
    A[4, :] = [2, 4, 3, 1, 5]  # jiaxin
    A[5, :] = [4, 3, 2, 1, 5]  # wenyang
    A[6, :] = [3, 4, 5, 1, 2]  # lixiao
    A[7, :] = [2, 4, 3, 1, 5]  # gaopeng
    A[8, :] = [3, 5, 1, 4, 2]  # minjun

    return A

args = vote()
A = para_ini()
score = np.zeros((1, 5))

for p in range(5):
    #print('p=', p)
    compare = [1, 2, 3, 4, 5]
    opt1 = compare[p]
    compare.pop(p)
    #print('opt1=', opt1)
    #print('compare=', compare)
    for j in range(4):
        #print('j=', j)
        opt2 = compare[j]
        #print('opt2=', opt2)
        count = 0
        for i in range(args.N):
            B = A[i, :]
            #print('B=', B)
            for q in range(5):
                if B[q] == opt1:
                    count += 1
                    break
                elif B[q] == opt2:
                    count += 0
                    break
        #print('count=', count)
        if count >= 5:
            score[0, p] += 1
            #print('score[0,p]=', score[0, p])
        #print('#############')

#print('count=', count)
#print('score=', score)
'''

'''
class Solution(object):
  def largeGroupPositions(self, s:str):
    res = []
    cur_res = []
    n = len(s)
    i = 0
    while (i < n):
      left = i
      right = i + 1
      while (right < n and s[left] == s[right]):
        right += 1

      if ((right - left) >= 3):
        res.add(left)
        res.add(right - 1)
        cur_res.add(res)
      i = right
    return cur_res

s = "abbxxxxzzy"
largeGroupPositions(s)
'''
