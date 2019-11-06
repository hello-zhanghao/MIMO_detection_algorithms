import tensorflow as tf 

class Detector(object):
    def __init__(self, params):
        self.Nr = params['Nr']
        self.Nt = params['Nt']
        self.H = tf.compat.v1.placeholder(shape=[None, 2*self.Nr, 2*self.Nt], dtype=tf.float32)
        self.y = tf.compat.v1.placeholder(shape=[None, 2*self.Nr], dtype=tf.float32)
        self.x = tf.compat.v1.placeholder(shape=[None, 2*self.Nt], dtype=tf.float32)
        self.noise_sigma2 = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
        self.batch_size = tf.shape(self.H)[0]
        self.L = params['OAMP_layer']
        self.constellation = params['constellation']
        self.M = tf.shape(params['constellation'])[0]
        self.learning_rate = params['learning_rate']

    def OAMP(self):
        """
        OAMP:
        """
        xhatk = tf.zeros(shape=[self.batch_size, 2*self.Nt], dtype=tf.float32)
        rk = self.y
        xhat = []
        for k in range(1, self.L+1):
            xhatk, rk = self.layer(xhatk, rk)
            xhat.append(xhatk)
        x_hatk_idx = self.demodulate(xhatk)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, x_hatk_idx)
        ser = 1-accuracy
        OAMP_nodes={
            'ser': ser,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
        }
        return OAMP_nodes

    def layer(self, xhatt, rt):
        HTH = tf.matmul(self.H, self.H, transpose_a=True)
        HHT = tf.matmul(self.H, self.H, transpose_b=True)
        v2_t = tf.divide(tf.reduce_sum(tf.square(rt), axis=1, keepdims=True)-tf.cast(
            2*self.Nr, tf.float32)*self.noise_sigma2/2., tf.expand_dims(tf.linalg.trace(HTH), axis=1))
        v2_t = tf.maximum(v2_t, 1e-9)
        v2_t = tf.expand_dims(v2_t, axis=1)
        lam_t = tf.expand_dims(self.noise_sigma2, axis=2)/2.  # shape=(1,1,1)
        inv_term_t = tf.linalg.inv(v2_t*HHT +
                                   lam_t*tf.eye(2*self.Nr, batch_shape=[self.batch_size]))  # shape=(?,2*Nr,2*Nr )
        What_t = v2_t * tf.matmul(self.H, inv_term_t, transpose_a=True)  # shape=(?,2*Nt,2*Nr)
        W_t = tf.cast(2*self.Nt, tf.float32)*What_t / tf.reshape(tf.linalg.trace(tf.matmul(What_t, self.H)), [-1, 1, 1])
        z_t = xhatt + self.batch_matvec_mul(W_t, rt)
        B_t = tf.eye(2*self.Nt, batch_shape=[self.batch_size])-tf.matmul(W_t, self.H)
        tau2_t = 1./tf.cast(2*self.Nt, tf.float32) * tf.reshape(
            tf.linalg.trace(tf.matmul(B_t, B_t, transpose_b=True)), [-1, 1, 1]) * v2_t \
                 + 1./tf.cast(2*self.Nt, tf.float32)*tf.reshape(self.noise_sigma2, [-1, 1, 1])\
                 *tf.reshape(tf.linalg.trace(tf.matmul(W_t, W_t, transpose_b=True)), [-1, 1, 1])
        xhatt= self.gaussian(z_t, tau2_t) 
        rt = self.y - self.batch_matvec_mul(self.H, xhatt)
        return xhatt, rt

    def gaussian(self, zt, tau2_t):
        tau2_t = tau2_t
        arg = tf.reshape(zt, [-1, 1]) - self.constellation
        arg = tf.reshape(arg, [-1, 2*self.Nt, self.M])
        arg = - (tf.square(arg) / 2. / tau2_t)
        arg = tf.reshape(arg, [-1, self.M])
        shatt1 = tf.nn.softmax(arg, axis=1)
        shatt1 = tf.matmul(shatt1, tf.reshape(self.constellation, [self.M, 1]))
        shatt1 = tf.reshape(shatt1, [-1, 2*self.Nt])
        return shatt1
    
    def batch_matvec_mul(self, A, b, transpose_a = False):
        """
        矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
        b.shape = (batch_size, Nt)
        输出矩阵C，C.shape = (batch_size, Nr)
        """
        C = tf.matmul(A, tf.expand_dims(b, axis=2), transpose_a=transpose_a)
        return tf.squeeze(C, -1)

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
    
    def accuracy(self, x, y):
        """
        Computes the fraction of elements for which x and y are equal
        """
        return tf.reduce_mean(tf.cast(tf.equal(x, y), tf.float32))

    def MMSE(self):
        """
        最小均方误差检测算法
        Input:
        y:接收信号 Tensor(shape=(batchsize, 2*Nr), dtype=float32)
        H:信道矩阵 Tensor(shape=(batchize, 2*Nr, 2*Nt), dtype=float32)
        noise_sigma2:噪声方差 Tensor(shape=(batchsize, 1), dtype=float32)
        Output:
        x: MMSE检测输出 Tensor(shape=(batch_size, Nt)
        """
        # Projected channel output
        Hty = self.batch_matvec_mul(tf.transpose(self.H, perm=[0, 2, 1]), self.y)

        # Gramian of transposed channel matrix
        HtH = tf.matmul(self.H, self.H, transpose_a=True)
        
        # Inverse Gramian
        HtHinv = tf.linalg.inv(
            HtH + tf.reshape(self.noise_sigma2/2, [-1, 1, 1]) * tf.eye(2*self.Nt, batch_shape=[self.batch_size]))

        # MMSE Detector
        xhat = self.batch_matvec_mul(HtHinv, Hty)
        xhat_idx = self.demodulate(xhat)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, xhat_idx)
        ser = 1-accuracy
        MMSE_nodes={
            'ser': ser,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
        }
        return MMSE_nodes
    
    def ZF(self):
        """
        迫零检测算法
        Input:
        y:接收信号 Tensor(shape=(batchsize, 2*Nr), dtype=float32)
        H:信道矩阵 Tensor(shape=(batchize, 2*Nr, 2*Nt), dtype=float32)
        noise_sigma2:噪声方差 Tensor(shape=(batchsize, 1), dtype=float32)
        Output:
        x: MMSE检测输出 Tensor(shape=(batch_size, Nt)
        """
        # Projected channel output
        Hty = self.batch_matvec_mul(tf.transpose(self.H, perm=[0, 2, 1]), self.y)

        # Gramian of transposed channel matrix
        HtH = tf.matmul(self.H, self.H, transpose_a=True)
        
        # Inverse Gramian
        HtHinv = tf.linalg.inv(HtH)

        # MMSE Detector
        xhat = self.batch_matvec_mul(HtHinv, Hty)
        xhat_idx = self.demodulate(xhat)
        indices = self.demodulate(self.x)
        accuracy = self.accuracy(indices, xhat_idx)
        ser = 1-accuracy
        ZF_nodes={
            'ser': ser,
            'x': self.x,
            'y': self.y,
            'H': self.H,
            'noise_sigma2': self.noise_sigma2,
            'M': self.M,
        }
        return ZF_nodes

    def SIC(self):
        H = self.H
        y = self.y
        for i in range(self.Nt*2):
            HTH = tf.matmul(H, H, transpose_a=True)
            W = tf.matmul(tf.linalg.inv(HTH), H, transpose_b=True)
            W_pow = tf.reduce_sum(W*W, axis=2)
            W_index = tf.argmin(W_pow, axis=1)
            xhat = self.batch_matvec_mul(W, y)


