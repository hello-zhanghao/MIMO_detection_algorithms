import tensorflow as tf


class DetNetDemod(object):
    def __init__(self, params):
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.H = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr, 2 * self.Nt], dtype=tf.float32)
        self.y = tf.compat.v1.placeholder(shape=[None, 2 * self.Nr], dtype=tf.float32)
        self.x = tf.compat.v1.placeholder(shape=[None, 2 * self.Nt], dtype=tf.float32)
        self.noise_sigma2 = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
        self.batch_size = tf.shape(self.H)[0]
        self.L = params['DetNetDemod_layer']
        self.constellation = params['constellation']
        self.M = tf.shape(params['constellation'])[0]
        self.learning_rate = params['learning_rate']

    def creat_graph(self, ZF_initial=True):
        # ZF_initial
        if ZF_initial:
            Hty = self.batch_matvec_mul(self.H, self.y, transpose_a=True)
            HtH = tf.matmul(self.H, self.H, transpose_a=True)
            HtHinv = tf.linalg.inv(HtH)
            xhatk = self.batch_matvec_mul(HtHinv, Hty)
        else:
            xhatk = tf.zeros(shape=[self.batch_size, 2 * self.Nt], dtype=tf.float32)

        xhat = []
        for k in range(1, self.L + 1):
            xhatk = self.layer(xhatk)
            if k % 5 == 0:
                xhatk = self.demod2constellation(xhatk)
            xhat.append(xhatk)
        print("Total number of trainable variables in DetNet", self.get_n_vars())
        loss = self.loss_fun(xhat, self.x)
        train_step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(tf.reduce_mean(loss))
        x_hatk_idx = self.demodulate(xhatk)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, x_hatk_idx)
        ser = 1 - accuracy
        DetNet_nodes = {
            'xhat': xhatk,
            'ser': ser,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
            'train': train_step,
            'loss': loss,
            'indices': indices,
        }
        return DetNet_nodes

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

    def layer(self, xhatt):
        # 线性
        W = tf.compat.v1.Variable(tf.random.normal(shape=[1, 2*self.Nt, 6*self.Nt], mean=0.0, stddev=0.01))
        b = tf.compat.v1.Variable(tf.constant(0.001, shape=(1, 2*self.Nt)), dtype=tf.float32)
        W_t = tf.tile(W, [self.batch_size, 1, 1])  # shape=(batch_size,2*Nt,  6*Nt)
        b_t = tf.tile(b, [self.batch_size, 1])
        HTH = tf.matmul(self.H, self.H, transpose_a=True)
        HTy = self.batch_matvec_mul(self.H, self.y, transpose_a=True)  # shape=(batch_size,2*Nt)
        HTHxhat = self.batch_matvec_mul(HTH, xhatt)  # shape=(batch_size,2*Nt)
        concatenation = tf.concat([HTy, xhatt, HTHxhat], axis=1)  # shape=(batch_size, 6*Nt)
        cal_t = self.batch_matvec_mul(W_t, concatenation) + b_t  # shape=(batch_szie, 2*Nt)

        # 非线性
        t = tf.compat.v1.Variable(0.5, dtype=tf.float32)
        xhatt = -1.0 + tf.nn.relu(cal_t + t)/tf.abs(t) - tf.nn.relu(cal_t - t)/tf.abs(t)
        return xhatt

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
        x_complex = tf.complex(x[:, 0: self.Nt], x[:, self.Nt: self.Nt*2])
        x_complex = tf.reshape(x_complex, shape=[-1, 1])
        constellation = tf.reshape(self.constellation, [1, -1])
        constellation_complex = tf.reshape(tf.complex(constellation, 0.)
                                           - tf.complex(0., tf.transpose(constellation)), [1, -1])
        indices = tf.cast(tf.argmin(tf.abs(x_complex - constellation_complex), axis=1), tf.int32)
        indices = tf.reshape(indices, shape=tf.shape(x_complex))
        return indices


    def demod2constellation(self, x_EST):
        shape = tf.shape(x_EST)
        x_EST = tf.reshape(x_EST, shape=[-1, 1])
        constellation = tf.reshape(self.constellation, [1, -1])
        indices = tf.cast(tf.argmin(tf.abs(x_EST - constellation), axis=1), tf.int32)
        indices = tf.reshape(indices, shape=shape)
        x_EST = tf.gather(self.constellation, indices)  # x_EST是解调后的信号
        return x_EST





