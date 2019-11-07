# coding:utf-8
# python 3.7 tensorflow1.14

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
import datetime

from Dataset import genData
from DetNetSIC3 import DetNetSIC3

from utils import model_eval, model_train, model_loss, model_est


dt = time.localtime()
ft = '%Y%m%d%H%M%S'
nt = time.strftime(ft, dt)
print('==============当前时间：%s===============' % nt)

# 消除信息 KMP_AFFINITY: pid 13764 tid 13280 thread 1 bound to OS proc set 1
os.environ['KMP_WARNINGS'] = 'off'
# 手动设置参数
# 测试DetNetSIC3,每一步分开训练
params = {
    # 二选一
    'dataset_dir': r'D:\Nr8Nt8batch_size500mod_nameQAM_4',  # 使用固定数据集
    # 'dataset_dir': None,  # 程序运行时生成数据集

    # ************************程序运行之前先检查下面的参数*****************
    # 仿真参数
    'mod_name': 'QAM_4',
    'constellation': np.array([0.7071, -0.7071], dtype=np.float32),
    'Nt': 8,  # Number of transmit antennas
    'Nr': 8,  # Number of transmit antennas
    'batch_size': 500,
    'DetNetSIC3_ZF_initial': True,
    'DetNetSIC3_layer': 30,
    'DetNetSIC3_outuser': 2,
    # 训练网络时的信噪比，一般选择误符号率在1e-2左右
    'SNR_dB_train': 20,
    # 网络的学习速率，手动调节
    'learning_rate': 0.001,  # Learning rate
    # 网络训练的迭代次数
    'maxEpoch': 10000,
    'nRounds': 1,
    # 测试检测算法的信噪比，一般误符号率到1e-4就行
    'SNR_dB_min_test': 0,  # Minimum SNR value in dB for simulation
    'SNR_dB_max_test': 40,  # Maximum SNR value in dB for simulation
    'SNR_step_test': 5,
    # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上
    'test_iterations': 200,
    # 模型，数据，图片保存位置
    'model_savedir': 'model'+nt,
    'results_savedir': r'.\results',
    'figures_savedir': r'.\figures',
    }
print(params)

# 数据生成对象
gen_data = genData(params)

DetNetSIC3_node1, DetNetSIC3_node2, DetNetSIC3_node3, DetNetSIC3_node4, DetNetSIC3_nodes = DetNetSIC3(params).creat_graph(ZF_initial=params['DetNetSIC3_ZF_initial'])

init = tf.compat.v1.global_variables_initializer()

# Create a session to run Tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session(config=config)
sess.run(init)

# Training
print('=========== is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
train_st = time.time()
for n in range(params['nRounds']):
    for epoch in range(params['maxEpoch']):
        x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = gen_data.dataTrain(epoch, params['SNR_dB_train'])
        data_Feed = {
            'x': x_Feed,
            'y': y_Feed,
            'H': H_Feed,
            'noise_sigma2': noise_sigma2_Feed,
        }
        model_train(sess, DetNetSIC3_node1, data_Feed)
        if epoch % 100 == 0:
            print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                  datetime.datetime.now().strftime('%X')))
            ser, loss = model_loss(sess, DetNetSIC3_node1, data_Feed)
            print('ser', ser, 'loss=', loss, '\n')
train_ed = time.time()
print("Train time is: "+str(train_ed-train_st))

train_st = time.time()
for n in range(params['nRounds']):
    for epoch in range(params['maxEpoch']):
        x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = gen_data.dataTrain(epoch, params['SNR_dB_train'])
        data_Feed = {
            'x': x_Feed,
            'y': y_Feed,
            'H': H_Feed,
            'noise_sigma2': noise_sigma2_Feed,
        }
        model_train(sess, DetNetSIC3_node2, data_Feed)
        if epoch % 100 == 0:
            print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                  datetime.datetime.now().strftime('%X')))
            ser, loss = model_loss(sess, DetNetSIC3_node2, data_Feed)
            print('ser', ser, 'loss=', loss, '\n')
train_ed = time.time()
print("Train time is: "+str(train_ed-train_st))

train_st = time.time()
for n in range(params['nRounds']):
    for epoch in range(params['maxEpoch']):
        x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = gen_data.dataTrain(epoch, params['SNR_dB_train'])
        data_Feed = {
            'x': x_Feed,
            'y': y_Feed,
            'H': H_Feed,
            'noise_sigma2': noise_sigma2_Feed,
        }
        model_train(sess, DetNetSIC3_node3, data_Feed)
        if epoch % 100 == 0:
            print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                  datetime.datetime.now().strftime('%X')))
            ser, loss = model_loss(sess, DetNetSIC3_node3, data_Feed)
            print('ser', ser, 'loss=', loss, '\n')
train_ed = time.time()
print("Train time is: "+str(train_ed-train_st))

train_st = time.time()
for n in range(params['nRounds']):
    for epoch in range(params['maxEpoch']):
        x_Feed, H_Feed, y_Feed, noise_sigma2_Feed = gen_data.dataTrain(epoch, params['SNR_dB_train'])
        data_Feed = {
            'x': x_Feed,
            'y': y_Feed,
            'H': H_Feed,
            'noise_sigma2': noise_sigma2_Feed,
        }
        model_train(sess, DetNetSIC3_node4, data_Feed)
        if epoch % 100 == 0:
            print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                  datetime.datetime.now().strftime('%X')))
            ser, loss = model_loss(sess, DetNetSIC3_node4, data_Feed)
            print('ser', ser, 'loss=', loss, '\n')
train_ed = time.time()
print("Train time is: "+str(train_ed-train_st))


# Testing
results = {
    'SNR_dBs': np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])}
sers = model_eval(sess, params, DetNetSIC3_nodes, gen_data, params['test_iterations'])
results['ser_DetNetSIC3s'] = sers
print(sers)

# plot
for key, value in results.items():
    if key == 'SNR_dBs':
        pass
    else:
        print(key, value)
        plt.plot(results['SNR_dBs'], value, label=key)
plt.grid(True, which='minor', linestyle='--')
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('SER')
plt.title('Nr%dNt%d_mod%s' % (params['Nr'], params['Nt'], params['mod_name']))
plt.legend()
plt.show()

