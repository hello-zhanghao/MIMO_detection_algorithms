import numpy as np
import scipy.io as scio

class genData(object):
    def __init__(self, params):
        self.dataset_dir = params['dataset_dir']
        self.batch_size = params['batch_size']
        self.constellation = params['constellation']
        self.Nr = params['Nr']
        self.Nt = params['Nt']
    
    def dataTrain(self, number, snr):
        if self.dataset_dir:
            data_train = scio.loadmat(self.dataset_dir+'\\train_data%d' % number)
            x = data_train['x']
            y = data_train['y']
            H = data_train['H']
            sigma2 = self.Nt / (np.power(10, snr/10) * self.Nr)
            noise = np.sqrt(sigma2/2)*(np.random.randn(x.shape[0], self.Nr)
                                       + 1j*np.random.randn(x.shape[0], self.Nr))
            y_noise = y + noise
            x_real = self.complex2real(x, matrix=False)
            H_real = self.complex2real(H, matrix=True)
            y_noise_real = self.complex2real(y_noise, matrix=False)
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [x.shape[0], 1])
        else:
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, 2 * self.Nt])
            x_real = self.constellation[s]
            # H_real = np.random.randn(self.batch_size, 2*self.Nr, 2*self.Nt) * np.sqrt(0.5 / self.Nr)  # 直接这样生成容易导致HTH不可逆,不知道为啥
            Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
            y_real = self.batch_matvec_mul(H_real, x_real)
            # power_rx = np.mean(np.sum(np.square(y_real), axis=1), axis=0)
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])
        return x_real, H_real, y_noise_real, sigma2

    def dataTest(self, number, snr):
        if self.dataset_dir:
            data_train = scio.loadmat(self.dataset_dir+'\\test_data%d' % number)
            x = data_train['x']
            y = data_train['y']
            H = data_train['H']
            sigma2 = self.Nt / (np.power(10, snr/10) * self.Nr)
            noise = np.sqrt(sigma2/2)*(np.random.randn(x.shape[0], self.Nr)
                                       + 1j*np.random.randn(x.shape[0], self.Nr))
            y_noise = y + noise
            x_real = self.complex2real(x, matrix=False)
            H_real = self.complex2real(H, matrix=True)
            y_noise_real = self.complex2real(y_noise, matrix=False)
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [x.shape[0], 1])
        else:
            s = np.random.randint(low=0, high=np.shape(self.constellation)[0], size=[self.batch_size, 2 * self.Nt])
            x_real = self.constellation[s]
            # H_real = np.random.randn(self.batch_size, 2 * self.Nr, 2 * self.Nt) * np.sqrt(0.5 / self.Nr)
            Hr = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            Hi = np.random.randn(self.batch_size, self.Nr, self.Nt) * np.sqrt(0.5 / self.Nr)
            H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
            y_real = self.batch_matvec_mul(H_real, x_real)
            # power_rx = np.mean(np.sum(np.square(y_real), axis=1), axis=0)
            sigma2 = self.Nt / (np.power(10, snr / 10) * self.Nr)
            noise = np.sqrt(sigma2 / 2) * np.random.randn(self.batch_size, 2 * self.Nr)
            y_noise_real = y_real + noise
            sigma2 = np.tile(np.expand_dims(np.expand_dims(sigma2, axis=0), axis=1), [self.batch_size, 1])
        return x_real, H_real, y_noise_real, sigma2

    def complex2real(self, x, matrix=True):
        if matrix:
            H1 = np.concatenate([np.real(x), -np.imag(x)], axis=2)
            H2 = np.concatenate([np.imag(x), np.real(x)], axis=2)
            H = np.concatenate([H1, H2], axis=1)
        else:
            H = np.concatenate([np.real(x), np.imag(x)], axis=1)
        return H

    def batch_matvec_mul(self, A, b):
        """
        矩阵A与矩阵b相乘，其中A.shape=(batch_size, Nr, Nt)
        b.shape = (batch_size, Nt)
        输出矩阵C，C.shape = (batch_size, Nr)
        """
        C = np.matmul(A, np.expand_dims(b, axis=2))
        return np.squeeze(C, -1)
