import tensorflow as tf
import numpy as np


class DetNetSIC3(object):
    def __init__(self, params):
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.H = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr, 2 * self.Nt], dtype=tf.float32)
        self.y = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr], dtype=tf.float32)
        self.x = tf.compat.v1.placeholder(shape=[None, 2 * self.Nt], dtype=tf.float32)
        self.noise_sigma2 = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
        self.batch_size = tf.shape(self.H)[0]
        self.L = params['DetNetSIC3_layer']
        self.outuser = params['DetNetSIC3_outuser']
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
        xhat_cut, xhat, train_vars = self.SIC(xhatk_initial, self.H, self.y)

        first_vars = train_vars[0]
        target_user = np.reshape(np.concatenate(([range(0, 2)],
                                                 [range(0 + self.Nt, 2 + self.Nt)]), axis=1), [-1])
        target_user = tf.cast(target_user, tf.int32)
        x_est1 = xhat_cut[0][-1]
        print(111, xhat_cut[1])
        print(222, tf.gather(self.x, target_user, axis=1))
        loss1 = self.loss_fun(xhat_cut[0], tf.gather(self.x, target_user, axis=1))
        train_step1 = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(tf.reduce_mean(loss1), var_list=first_vars)
        x_est_idx1 = self.demodulate(x_est1)
        indices1 = self.demodulate(tf.gather(self.x, target_user, axis=1))
        accuracy1 = self.accuracy(indices1, x_est_idx1)
        ser1 = 1 - accuracy1
        DetNetSIC3_node1 = {
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

        second_vars = train_vars[1]
        for variable in train_vars[0]:
            second_vars.remove(variable)
        target_user = np.reshape(np.concatenate(([range(2, 4)],
                                                 [range(2 + self.Nt, 4 + self.Nt)]), axis=1), [-1])
        target_user = tf.cast(target_user, tf.int32)
        x_est2 = xhat_cut[1][-1]

        loss2 = self.loss_fun(xhat_cut[1], tf.gather(self.x, target_user, axis=1))
        train_step2 = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(tf.reduce_mean(loss2), var_list=second_vars)
        x_est_idx2 = self.demodulate(x_est2)
        indices2 = self.demodulate(tf.gather(self.x, target_user, axis=1))
        accuracy2 = self.accuracy(indices2, x_est_idx2)
        ser2 = 1 - accuracy2
        DetNetSIC3_node2 = {
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

        third_vars = train_vars[2]
        for variable in train_vars[1]:
            third_vars.remove(variable)
        target_user = np.reshape(np.concatenate(([range(4, 6)],
                                                 [range(4 + self.Nt, 6 + self.Nt)]), axis=1), [-1])
        target_user = tf.cast(target_user, tf.int32)
        x_est3 = xhat_cut[2][-1]
        loss3 = self.loss_fun(xhat_cut[2], tf.gather(self.x, target_user, axis=1))
        train_step3 = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(tf.reduce_mean(loss3), var_list=third_vars)
        x_est_idx3 = self.demodulate(x_est3)
        indices3 = self.demodulate(tf.gather(self.x, target_user, axis=1))
        accuracy3 = self.accuracy(indices3, x_est_idx3)
        ser3 = 1 - accuracy3
        DetNetSIC3_node3 = {
            'xhat': x_est3,
            'ser': ser3,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
            'train': train_step3,
            'loss': loss3,
        }

        forth_vars = train_vars[3]
        for variable in train_vars[2]:
            forth_vars.remove(variable)
        target_user = np.reshape(np.concatenate(([range(6, 8)],
                                                 [range(6 + self.Nt, 8 + self.Nt)]), axis=1), [-1])
        target_user = tf.cast(target_user, tf.int32)
        x_est4 = xhat_cut[3][-1]
        loss4 = self.loss_fun(xhat_cut[3], tf.gather(self.x, target_user, axis=1))
        train_step4 = tf.compat.v1.train.AdamOptimizer(
            self.learning_rate).minimize(tf.reduce_mean(loss4), var_list=forth_vars)
        x_est_idx4 = self.demodulate(x_est4)
        indices4 = self.demodulate(tf.gather(self.x, target_user, axis=1))
        accuracy4 = self.accuracy(indices4, x_est_idx4)
        ser4 = 1 - accuracy4
        DetNetSIC3_node4 = {
            'xhat': x_est4,
            'ser': ser4,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
            'train': train_step4,
            'loss': loss4,
        }

        x_est = xhat[-1]
        x_est_idx = self.demodulate(x_est)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, x_est_idx)
        ser = 1 - accuracy
        DetNetSIC3_nodes = {
            'xhat': x_est,
            'ser': ser,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
        }
        return DetNetSIC3_node1, DetNetSIC3_node2, DetNetSIC3_node3, DetNetSIC3_node4, DetNetSIC3_nodes

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
            H_cut = tf.gather(H, target_user, axis=2)
            x_cut = tf.gather(xhatt, target_user, axis=1)
            xhat_cut[i] = self.DetNet(x_cut, H_cut, y_cut, self.L, self.outuser)
        xhat = xhat_cut[0]
        print('4', xhat[0])
        for i in range(self.L):
            for j in range(1, self.Nt // self.outuser):
                Nt_temp = xhat[i].shape.as_list()[1]
                xhat[i] = tf.concat([xhat[i][:, 0:Nt_temp//2], xhat_cut[j][i][:, 0:self.outuser],
                                    xhat[i][:, Nt_temp//2:Nt_temp],
                                     xhat_cut[j][i][:, self.outuser:self.outuser*2]], axis=1)
        return xhat

    def SIC(self, xhatt, H, y):
        train_vars = []
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
            H_cut = tf.gather(H, target_user, axis=2)
            x_cut = tf.gather(xhatt, target_user, axis=1)
            xhat_cut[i] = self.DetNet(x_cut, H_cut, y_cut, self.L, self.outuser)
            print(i, xhat_cut[i])
            train_vars.append(tf.compat.v1.trainable_variables())  # 当前可训练变量集合

            x_EST = xhat_cut[i][-1]
            shape = tf.shape(x_EST)
            x_EST = tf.reshape(x_EST, shape=[-1, 1])
            constellation = tf.reshape(self.constellation, [1, -1])
            indices = tf.cast(tf.argmin(tf.abs(x_EST - constellation), axis=1), tf.int32)
            indices = tf.reshape(indices, shape=shape)
            x_EST = tf.gather(self.constellation, indices)  # x_EST是解调后的信号
            # print('1', x_EST)
            xhatt = tf.concat([xhatt[:, 0:start],
                               x_EST[:, 0:self.outuser],
                               xhatt[:, end:self.Nt+start],
                               x_EST[:, self.outuser: self.outuser*2],
                               xhatt[:, self.Nt+end: self.Nt+self.Nt]], axis=1)
        xhat = xhat_cut[0].copy()
        # print('4', xhat[0])
        for i in range(self.L):
            for j in range(1, self.Nt // self.outuser):
                Nt_temp = xhat[i].shape.as_list()[1]
                xhat[i] = tf.concat([xhat[i][:, 0:Nt_temp//2], xhat_cut[j][i][:, 0:self.outuser],
                                    xhat[i][:, Nt_temp//2:Nt_temp],
                                     xhat_cut[j][i][:, self.outuser:self.outuser*2]], axis=1)
        return xhat_cut, xhat, train_vars

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




