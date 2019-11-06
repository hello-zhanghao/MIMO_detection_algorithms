import tensorflow as tf
import numpy as np


class DetNetPIC(object):
    def __init__(self, params):
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.H = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr, 2 * self.Nt], dtype=tf.float32)
        self.y = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr], dtype=tf.float32)
        self.x = tf.compat.v1.placeholder(shape=[None, 2 * self.Nt], dtype=tf.float32)
        self.noise_sigma2 = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
        self.batch_size = tf.shape(self.H)[0]
        self.L1 = params['DetNetPIC_layer1']
        self.L2 = params['DetNetPIC_layer2']
        self.outuser = params['DetNetPIC_outuser']
        self.constellation = params['constellation']
        self.M = tf.shape(params['constellation'])[0]
        self.learning_rate = params['learning_rate']

    def creat_graph(self, ZF_initial=True):
        # ZF initial
        if ZF_initial:
            Hty = self.batch_matvec_mul(self.H, self.y, transpose_a=True)
            HtH = tf.matmul(self.H, self.H, transpose_a=True)
            HtHinv = tf.linalg.inv(HtH)
            xhatk_initial = self.batch_matvec_mul(HtHinv, Hty)
        else:
            xhatk_initial = tf.zeros(shape=[self.batch_size, 2 * self.Nt], dtype=tf.float32)

        # 第一级 DetNet预估计
        xhat1 = self.DetNet(xhatk_initial, self.H, self.y, self.L1, self.Nt)
        x_est1 = xhat1[-1]
        first_train_vars = tf.compat.v1.trainable_variables()  # 当前可训练变量集合
        print('first_train_vars', first_train_vars, '\n', len(first_train_vars))
        print("Total number of trainable variables in DetNetPIC1", self.get_n_vars())
        loss1 = self.loss_fun(xhat1, self.x)
        train_step1 = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(tf.reduce_mean(loss1))
        x_est1_idx = self.demodulate(x_est1)
        indices1 = self.demodulate(self.x)
        accuracy1 = self.accuracy(indices1, x_est1_idx)
        ser1 = 1 - accuracy1

        # 第二级 干扰消除
        # 1 将估计符号解调到相应的星座点上， 用于干扰消除
        X_EST = x_est1
        shape = tf.shape(X_EST)
        X_EST = tf.reshape(X_EST, shape=[-1, 1])
        constellation = tf.reshape(self.constellation, [1, -1])
        indices = tf.cast(tf.argmin(tf.abs(X_EST-constellation), axis=1), tf.int32)
        indices = tf.reshape(indices, shape=shape)
        X_EST = tf.gather(self.constellation, indices)  # X_EST是解调后的信号

        # 2 并行干扰消除
        xhat2 = self.PIC(X_EST, self.H, self.y)
        x_est2 = xhat2[-1]
        all_train_vars = tf.compat.v1.trainable_variables()
        second_train_vars = all_train_vars.copy()
        for variable in first_train_vars:
            second_train_vars.remove(variable)
        print('second_train_vars', second_train_vars, '\n', len(second_train_vars))
        print('all_train_vars', all_train_vars, '\n', len(all_train_vars))
        print("Total number of trainable variables in DetNetPIC", self.get_n_vars())

        loss2 = self.loss_fun(xhat2, self.x)
        # train_step2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(
        #     tf.reduce_mean(loss2), var_list=second_train_vars)  # 冻结第一级DetNet变量
        train_step2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(
            tf.reduce_mean(loss2))  # 不冻结第一级DetNet变量
        x_est2_idx = self.demodulate(x_est2)
        indices2 = self.demodulate(self.x)
        accuracy2 = self.accuracy(indices2, x_est2_idx)
        ser2 = 1 - accuracy2
        DetNet2_nodes1 = {
            'xhat': x_est1,
            'ser': ser1,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
            'train': train_step1,
            'loss': loss1,
        }
        DetNet2_nodes2 = {
            'xhat': x_est2,
            'ser': ser2,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
            'train': train_step2,
            'loss': loss2,
        }
        return DetNet2_nodes1, DetNet2_nodes2

    def train(self, x_NN):
        loss = self.loss_fun(x_NN, self.x)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(tf.reduce_mean(loss))
        return loss, train_step

    def ser(self, xhat):
        x_hat_idx = self.demodulate(xhat)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, x_hat_idx)
        return accuracy

    def loss_fun(self, xhat, x):
        """
        损失函数1，每一层输出的x的估计xhatk与x的均方差之和
        Input:
        xhat: 每一层输出的x的估计，是一个包含L个元素的列表，每个元素是Tensor(shape=(batch_size, 2*Nt), dtype=tf.float32)
        x: 发送调制符号x Tensor(shape=(batch_size,2*Nt), dtype=float32)
        Output:
        loss: 损失值 Tensor(shape=(), dtype=float32)
        """
        loss = 0
        i = 0.0
        for xhatk in xhat:
            i += 1.0
            lk = tf.compat.v1.losses.mean_squared_error(labels=x, predictions=xhatk)
            loss += lk * tf.math.log(i)
            tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.LOSSES, lk)
        return loss

    def get_n_vars(self):
        total_parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters


    def DetNet(self, xhatt, H, y, L, Nt):
        xhat = []
        for i in range(L):
            W = tf.compat.v1.Variable(tf.random.normal(shape=[1, 2*Nt, 6*Nt], mean=0.0, stddev=0.01))
            b = tf.compat.v1.Variable(tf.constant(0.001, shape=[1, 2*Nt]), dtype=tf.float32)
            W_t = tf.tile(W, [tf.shape(xhatt)[0], 1, 1])
            b_t = tf.tile(b, [tf.shape(xhatt)[0], 1])
            HTH = tf.matmul(H, H, transpose_a=True)
            HTy = self.batch_matvec_mul(H, y, transpose_a=True)
            HTHxhatt = self.batch_matvec_mul(HTH, xhatt)
            concatenation = tf.concat([HTy, xhatt, HTHxhatt], axis=1)
            cal_t = self.batch_matvec_mul(W_t, concatenation) + b_t

            # 非线性
            t = tf.compat.v1.Variable(0.5, dtype=tf.float32)
            xhatt = -1.0 + tf.nn.relu(cal_t + t)/tf.abs(t) - tf.nn.relu(cal_t - t)/tf.abs(t)
            xhat.append(xhatt)
        return xhat

    def PIC(self, xhatt, H, y):
        xhat_cut = list(range(self.Nt // self.outuser))
        for i in range(self.Nt // self.outuser):
            start = i * self.outuser
            end = start + self.outuser
            target_user = np.reshape(np.concatenate(([range(start, end)],
                                                     [range(start + self.Nt, end + self.Nt)]), axis=1), [-1])
            cancel_user = np.reshape(np.concatenate(([range(0, start)], [range(end, self.Nt)],
                                                     [range(self.Nt, self.Nt + start)],
                                                     [range(self.Nt + end, self.Nt + self.Nt)]), axis=1), [-1])
            target_user = tf.cast(target_user, tf.int32)
            cancel_user = tf.cast(cancel_user, tf.int32)
            y_cut = y - self.batch_matvec_mul(tf.gather(H, cancel_user, axis=2), tf.gather(xhatt, cancel_user, axis=1))
            H_cut= tf.gather(H, target_user, axis=2)
            x_cut = tf.gather(xhatt, target_user, axis=1)
            xhat_cut[i] = self.DetNet(x_cut, H_cut, y_cut, self.L2, self.outuser)
        xhat = xhat_cut[0]
        print('4', xhat[0])
        for i in range(self.L2):
            for j in range(1, self.Nt // self.outuser):
                Nt_temp = xhat[i].shape.as_list()[1]
                xhat[i] = tf.concat([xhat[i][:, 0:Nt_temp//2], xhat_cut[j][i][:, 0:self.outuser],
                                    xhat[i][:, Nt_temp//2:Nt_temp],
                                     xhat_cut[j][i][:, self.outuser:self.outuser*2]], axis=1)
        return xhat

    def batch_matvec_mul(self, A, b, transpose_a=False):
        """
        矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
        b.shape = (batch_size, Nt)
        输出矩阵C，C.shape = (batch_size, Nr)
        """
        C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
        return tf.squeeze(C, -1)

    def accuracy(self, x, y):
        """
        Computes the fraction of elements for which x and y are equal
        """
        return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))

    def demodulate(self, x):
        """
        信号解调(复数域解调）
        Input:
        y: 检测器检测后的信号 Tensor(shape=(batchsize, 2*Nt), dtype=float32)
        constellation: 星座点，即发送符号可能的离散值 Tensor(shape=(np.sqrt(调制阶数), ), dtype=float32)
        Output:
        indices: 解调后的基带数据信号，Tensor(shape=(batchsize, Nt)， dtype=tf.int32)
        """
        Nt = x.shape.as_list()[1] // 2
        x_complex = tf.complex(x[:, 0: Nt], x[:, Nt: Nt*2])
        x_complex = tf.reshape(x_complex, shape=[-1, 1])
        constellation = tf.reshape(self.constellation, [1, -1])
        constellation_complex = tf.reshape(tf.complex(constellation, 0.)
                                           - tf.complex(0., tf.transpose(constellation)), [1, -1])
        indices = tf.cast(tf.argmin(tf.abs(x_complex - constellation_complex), axis=1), tf.int32)
        indices = tf.reshape(indices, shape=tf.shape(x_complex))
        return indices




