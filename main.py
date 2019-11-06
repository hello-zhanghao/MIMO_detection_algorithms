# coding:utf-8
# python 3.7 tensorflow1.14

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import time
import os
import datetime

from Dataset import genData
from OAMPNet import OAMPNet
from OAMPNet2 import OAMPNet2
from MMNet import MMNet
from DetNet import DetNet
from Detector import Detector
from DetNetPIC import DetNetPIC
from DetNetSIC import DetNetSIC
from DetNetDemod import DetNetDemod
from utils import model_eval, model_train, model_loss, model_est


dt = time.localtime()
ft = '%Y%m%d%H%M%S'
nt = time.strftime(ft, dt)
print('==============当前时间：%s===============' % nt)

# 消除信息 KMP_AFFINITY: pid 13764 tid 13280 thread 1 bound to OS proc set 1
os.environ['KMP_WARNINGS'] = 'off'
# 手动设置参数
########################## Read Me #######################
#所有仿真参数均在字典params中设置
# 1、savetype（列表）: 只有0是不保存任何东西, 有1是保存仿真结果误码率数据，有2是保存仿真结果误码率曲线，比如savetype=[1,2]是都保存
# 2、isTest(bool):True是测试不同算法的仿真结果，False是不测试任何算法
# 3、simulation_algorithms[列表]: 当isTest为真，且出现在该列表的算法才进行仿真
# 4
#########################################################
params = {
    # 二选一
    'dataset_dir': r'D:\Nr8Nt8batch_size500mod_nameQAM_4',  # 使用固定数据集
    # 'dataset_dir': None,  # 程序运行时生成数据集

    # ************************程序运行之前先检查下面的参数*****************
    'isTest': True,
    'isTrain': True,
    'savetype': [0],
    # 仿真算法
    'simulation_algorithms': [
        # 'MMNet',
        # 'OAMPNet2',
        # 'OAMPNet',
        'OAMP',
        # 'MMSE',
        'ZF',
        # 'DetNet',
        # 'DetNetPIC',
        # 'DetNetSIC',
        # 'DetNetDemod',
        ],
    # 仿真参数
    'mod_name': 'QAM_4',
    'constellation': np.array([0.7071, -0.7071], dtype=np.float32),
    'Nt': 8,  # Number of transmit antennas
    'Nr': 8,  # Number of transmit antennas
    'batch_size': 500,
    # 不同网络的层数
    'OAMP_layer': 10,
    'MMNet_layer': 10,
    'OAMPNet_layer': 10,
    'OAMPNet2_layer': 10,
    'DetNet_layer': 30,
    'DetNet_ZF_initial': False,
    'DetNetPIC_layer1': 30,
    'DetNetPIC_layer2': 30,
    'DetNetPIC_ZF_initial': False,
    'DetNetPIC_outuser': 2,
    'DetNetSIC_layer1': 30,
    'DetNetSIC_layer2': 30,
    'DetNetSIC_ZF_initial': False,
    'DetNetSIC_outuser': 2,
    'DetNetDemod_layer': 30,
    'DetNetDemod_ZF_initial': False,
    # 训练网络时的信噪比，一般选择误符号率在1e-2左右
    'SNR_dB_train': 20,
    # 网络的学习速率，手动调节
    'learning_rate': 0.001,  # Learning rate
    # 网络训练的迭代次数
    'maxEpoch': 10000,
    'nRounds': 20,
    # 测试检测算法的信噪比，一般误符号率到1e-4就行
    'SNR_dB_min_test': 0,  # Minimum SNR value in dB for simulation
    'SNR_dB_max_test': 40,  # Maximum SNR value in dB for simulation
    'SNR_step_test': 5,
    # 测试检测算法的迭代次数，一般保证最低误符号率的错误符号在1e2以上
    'test_iterations': 2000,
    # 模型，数据，图片保存位置
    'model_savedir': 'model'+nt,
    'results_savedir': r'.\results',
    'figures_savedir': r'.\figures',
    }
print(params)

# 数据生成对象
gen_data = genData(params)

# 建立传统方法计算图
detector = Detector(params)
OAMP_nodes = detector.OAMP()
MMSE_nodes = detector.MMSE()
ZF_nodes = detector.ZF()

# 建立OAMPNet2计算图
if 'OAMPNet2' in params['simulation_algorithms']:
    OAMPNet2 = OAMPNet2(params)
    OAMPNet2_nodes = OAMPNet2.creat_graph()

# # 建立OAMPNet计算图
if 'OAMPNet' in params['simulation_algorithms']:
    OAMPNet = OAMPNet(params)
    OAMPNet_nodes = OAMPNet.creat_graph()

# 建立MMNet计算图
if 'MMNet' in params['simulation_algorithms']:
    MMNet = MMNet(params)
    MMNet_nodes = MMNet.creat_graph()

# 建立DetNet计算图
if 'DetNet' in params['simulation_algorithms']:
    DetNet = DetNet(params)
    DetNet_nodes = DetNet.creat_graph(ZF_initial=params['DetNet_ZF_initial'])

# 建立DetNetPIC计算图
if 'DetNetPIC' in params['simulation_algorithms']:
    DetNetPIC = DetNetPIC(params)
    DetNetPIC_nodes1, DetNetPIC_nodes2 = DetNetPIC.creat_graph(ZF_initial=params['DetNetPIC_ZF_initial'])

# 建立DetNetSIC计算图
if 'DetNetSIC' in params['simulation_algorithms']:
    DetNetSIC = DetNetSIC(params)
    DetNetSIC_nodes1, DetNetSIC_nodes2 = DetNetSIC.creat_graph(ZF_initial=params['DetNetSIC_ZF_initial'])

# 建立DetNetDemod计算图
if 'DetNetDemod' in params['simulation_algorithms']:
    DetNetDemod = DetNetDemod(params)
    DetNetDemod_nodes = DetNetDemod.creat_graph(ZF_initial=params['DetNetDemod_ZF_initial'])


init = tf.compat.v1.global_variables_initializer()

# Create a session to run Tensorflow
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session(config=config)
sess.run(init)

# Training
loss_all = {}
if 'DetNet' in params['simulation_algorithms']:
    loss_DetNet = []
    print('===========DetNet is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNet_nodes, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNet, loss = model_loss(sess, DetNet_nodes, data_Feed)
                print('ser_DetNet', ser_DetNet, 'loss=', loss, '\n')
                loss_DetNet.append(loss)
    loss_all['DetNet'] = loss_DetNet  # 损失值
    train_ed = time.time()
    print("DetNet Train time is: "+str(train_ed-train_st))

if 'OAMPNet' in params['simulation_algorithms']:
    loss_OAMPNet = []
    print('===========OAMPNet is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, OAMPNet_nodes, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_OAMPNet, loss = model_loss(sess, OAMPNet_nodes, data_Feed)
                print('ser_OAMPNet', ser_OAMPNet, 'loss=', loss, '\n')
                loss_OAMPNet.append(loss)
    loss_all['OAMPNet'] = loss_OAMPNet
    train_ed = time.time()
    print("OAMPNet Train time is: "+str(train_ed-train_st))

if 'OAMPNet2' in params['simulation_algorithms']:
    loss_OAMPNet2 = []
    print('===========OAMPNet2 is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, OAMPNet2_nodes, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_OAMPNet2, loss = model_loss(sess, OAMPNet2_nodes, data_Feed)
                print('ser_OAMPNet2', ser_OAMPNet2, 'loss=', loss, '\n')
                loss_OAMPNet2.append(loss)
    loss_all['OAMPNet2'] = loss_OAMPNet2
    train_ed = time.time()
    print("OAMPNet2 Train time is: "+str(train_ed-train_st))

if 'MMNet' in params['simulation_algorithms']:
    loss_MMNet = []
    print('===========MMNet is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, MMNet_nodes, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_MMNet, loss = model_loss(sess, MMNet_nodes, data_Feed)
                print('ser_MMNet', ser_MMNet, 'loss=', loss, '\n')
                loss_MMNet.append(loss)
    loss_all['MMNet'] = loss_MMNet
    train_ed = time.time()
    print("MMNet Train time is: "+str(train_ed-train_st))

if 'DetNetPIC' in params['simulation_algorithms']:
    loss_DetNetPIC = []
    print('===========DetNetPIC1 is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNetPIC_nodes1, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNetPIC1, loss = model_loss(sess, DetNetPIC_nodes1, data_Feed)
                print('ser_DetNetPIC1', ser_DetNetPIC1, 'loss=', loss, '\n')
    train_ed = time.time()
    print("DetNetPIC1 Train time is: " + str(train_ed - train_st))

    print('===========DetNetPIC2 is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNetPIC_nodes2, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNetPIC2, loss = model_loss(sess, DetNetPIC_nodes2, data_Feed)
                print('ser_DetNetPIC2', ser_DetNetPIC2, 'loss=', loss, '\n')
                loss_DetNetPIC.append(loss)
    loss_all['DetNetPIC'] = loss_DetNetPIC
    train_ed = time.time()
    print("DetNetPIC2 Train time is: " + str(train_ed - train_st))

if 'DetNetSIC' in params['simulation_algorithms']:
    loss_DetNetSIC = []
    print('===========DetNetSIC1 is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNetSIC_nodes1, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNetSIC1, loss = model_loss(sess, DetNetSIC_nodes1, data_Feed)
                print('ser_DetNetSIC1', ser_DetNetSIC1, 'loss=', loss, '\n')
    train_ed = time.time()
    print("DetNetSIC1 Train time is: " + str(train_ed - train_st))

    print('===========DetNetSIC2 is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNetSIC_nodes2, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNetSIC2, loss = model_loss(sess, DetNetSIC_nodes2, data_Feed)
                print('ser_DetNetSIC2', ser_DetNetSIC2, 'loss=', loss, '\n')
                loss_DetNetSIC.append(loss)
    loss_all['DetNetSIC'] = loss_DetNetSIC
    train_ed = time.time()
    print("DetNetSIC2 Train time is: " + str(train_ed - train_st))

if 'DetNetDemod' in params['simulation_algorithms']:
    loss_DetNetDemod = []
    print('===========DetNetDemod is Training, now_time is %s=========' % (datetime.datetime.now().strftime('%X')))
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
            model_train(sess, DetNetDemod_nodes, data_Feed)
            if epoch % 1000 == 0:
                print('===========epoch%d,now_time is %s=========' % (epoch + n*params['maxEpoch'],
                                                                      datetime.datetime.now().strftime('%X')))
                ser_DetNetDemod, loss = model_loss(sess, DetNetDemod_nodes, data_Feed)
                print('ser_DetNetDemod', ser_DetNetDemod, 'loss=', loss, '\n')
                loss_DetNetDemod.append(loss)
    loss_all['DetNetDemod'] = loss_DetNetDemod  # 损失值
    train_ed = time.time()
    print("DetNetDemod Train time is: "+str(train_ed-train_st))

# Testing
results = {
    'SNR_dBs': np.arange(params['SNR_dB_min_test'], params['SNR_dB_max_test'], params['SNR_step_test'])}
if params['isTest']:
    test_st = time.time()
    if 'DetNetDemod' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('DetNetDemod', datetime.datetime.now().strftime('%X')))
        ser_DetNetDemods = model_eval(sess, params, DetNetDemod_nodes, gen_data, params['test_iterations'])
        results['ser_DetNetDemods'] = ser_DetNetDemods
        print(ser_DetNetDemods)

    if 'DetNetPIC' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('DetNetPIC', datetime.datetime.now().strftime('%X')))
        ser_DetNetPIC1s = model_eval(sess, params, DetNetPIC_nodes1, gen_data, params['test_iterations'])
        ser_DetNetPIC2s = model_eval(sess, params, DetNetPIC_nodes2, gen_data, params['test_iterations'])
        results['ser_DetNetPIC1s'] = ser_DetNetPIC1s
        results['ser_DetNetPIC2s'] = ser_DetNetPIC2s
        print(ser_DetNetPIC1s, '\n', ser_DetNetPIC2s)

    if 'DetNetSIC' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('DetNetSIC', datetime.datetime.now().strftime('%X')))
        ser_DetNetSIC1s = model_eval(sess, params, DetNetSIC_nodes1, gen_data, params['test_iterations'])
        ser_DetNetSIC2s = model_eval(sess, params, DetNetSIC_nodes2, gen_data, params['test_iterations'])
        results['ser_DetNetSIC1s'] = ser_DetNetSIC1s
        results['ser_DetNetSIC2s'] = ser_DetNetSIC2s
        print(ser_DetNetSIC1s, '\n', ser_DetNetSIC2s)

    if 'DetNet' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('DetNet', datetime.datetime.now().strftime('%X')))
        ser_DetNets = model_eval(sess, params, DetNet_nodes, gen_data, params['test_iterations'])
        results['ser_DetNets'] = ser_DetNets
        print(ser_DetNets)

    if 'OAMPNet' in params['simulation_algorithms']:
        print('=======正在测试%s检测方法, now_time is %s========' % ('OAMPNet', datetime.datetime.now().strftime('%X')))
        ser_OAMPNets = model_eval(sess, params, OAMPNet_nodes, gen_data, params['test_iterations'])
        results['ser_OAMPNets'] = ser_OAMPNets
        print(ser_OAMPNets)

    if 'OAMPNet2' in params['simulation_algorithms']:
        print('=======正在测试%s检测方法, now_time is %s=======' % ('OAMPNet2', datetime.datetime.now().strftime('%X')))
        ser_OAMPNet2s = model_eval(sess, params, OAMPNet2_nodes, gen_data, params['test_iterations'])
        results['ser_OAMPNet2s'] = ser_OAMPNet2s
        print(ser_OAMPNet2s)

    if 'MMNet' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s========' % ('MMNet', datetime.datetime.now().strftime('%X')))
        ser_MMNets = model_eval(sess, params, MMNet_nodes, gen_data, params['test_iterations'])
        results['ser_MMNets'] = ser_MMNets
        print(ser_MMNets)

    if 'OAMP' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('OAMP', datetime.datetime.now().strftime('%X')))
        ser_OAMPs = model_eval(sess, params, OAMP_nodes, gen_data, params['test_iterations'])
        results['ser_OAMPs'] = ser_OAMPs
        print(ser_OAMPs)

    if 'MMSE' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('MMSE', datetime.datetime.now().strftime('%X')))
        ser_MMSEs = model_eval(sess, params, MMSE_nodes, gen_data, params['test_iterations'])
        results['ser_MMSEs'] = ser_MMSEs
        print(ser_MMSEs)

    if 'ZF' in params['simulation_algorithms']:
        print('========正在测试%s检测方法, now_time is %s=========' % ('ZF', datetime.datetime.now().strftime('%X')))
        ser_ZFs = model_eval(sess, params, ZF_nodes, gen_data, params['test_iterations'])
        results['ser_ZFs'] = ser_ZFs
        print(ser_ZFs)
    test_ed = time.time()
    print("Test time is: "+str(test_ed-test_st))


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


# 保存结果
if 1 in params['savetype']:
    np.save(params['results_savedir']+'\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.npy'%(
            params['Nr'], params['Nt'], params['modulation'],params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), results)

# 保存图片
if 2 in params['savetype']:
    plt.savefig(params['figures_savedir']+'\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.jpg' % (
            params['Nr'], params['Nt'], params['modulation'], params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), format='jpg', dpi=1000)
    plt.savefig(params['figures_savedir']+'\\Nr%d_Nt%d_mod%s_SNR%d_%d_train_batchsize%d_maxEpoch%d_Layer%d_%s.eps' % (
            params['Nr'], params['Nt'], params['modulation'], params['SNR_dB_min'], params['SNR_dB_max'],
            params['train_batchsize'], params['maxEpoch'], params['L'], nt), format='eps', dpi=1000)
