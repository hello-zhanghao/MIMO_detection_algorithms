import tensorflow as tf  
import numpy as np


def model_train(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    sess.run(nodes['train'], feed_dict)

def model_loss(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    ser, loss = sess.run([nodes['ser'], nodes['loss']], feed_dict)
    return ser, loss

def model_est(sess, nodes, data):
    feed_dict = {
        nodes['x']: data['x'],
        nodes['y']: data['y'],
        nodes['H']: data['H'],
        nodes['noise_sigma2']: data['noise_sigma2'],
    }
    xhat= sess.run(nodes['xhat'], feed_dict)
    return xhat

def model_eval(sess, params, detector, Data, iterations=2000):
    SNR_dBs = np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])
    # print(SNR_dBs)
    ser_all = []
    for i in range(SNR_dBs.shape[0]):
        ser = 0.
        print('======================正在仿真SNR:%ddB================================' % (SNR_dBs[i]))
        for j in range(iterations):
            # 生成测试数据
            x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = Data.dataTest(j, SNR_dBs[i])
            # 信号检测
            feed_dict = {
                detector['x']: x_Feed,
                detector['y']: y_Feed,
                detector['H']: H_Feed,
                detector['noise_sigma2']: noise_sigma2_Feed,
                }
            ser += sess.run(detector['ser'], feed_dict) / iterations
        ser_all.append(ser)
    return ser_all


