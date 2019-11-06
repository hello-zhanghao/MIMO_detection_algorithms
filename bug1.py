# 如果使用tf.float32很容易出现矩阵不可逆的现象出项,大概100-200次就会出现,换成float64不会
# 2, 如果将矩阵H换成8*8然后按复数矩阵方式拼接也不会出问题
import numpy as np
import tensorflow as tf

H = tf.compat.v1.placeholder(shape=[500, 16, 16], dtype=tf.float32)
HTH = tf.matmul(H, H, transpose_a=True)
HTHinv = tf.linalg.inv(HTH)
sess = tf.Session()
for i in range(10000):
    print(i)
    # H =   # 直接这样生成容易导致HTH不可逆,不知道为啥
    Hr = np.random.randn(500, 8, 8) * np.sqrt(0.5 / 8)
    Hi = np.random.randn(500, 8, 8) * np.sqrt(0.5 / 8)
    H_real = np.concatenate([np.concatenate([Hr, -Hi], axis=2), np.concatenate([Hi, Hr], axis=2)], axis=1)
    # H_real = np.random.randn(500, 2*8, 2*8) * np.sqrt(0.5 / 8)
    sess.run(HTHinv, feed_dict={H: H_real})
sess.close()