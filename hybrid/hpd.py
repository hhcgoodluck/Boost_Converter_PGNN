import os
import scipy.io as spio
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adadelta, Adagrad, Adam, Nadam, SGD
from keras.callbacks import EarlyStopping, TerminateOnNaN
from keras import backend as K
from keras.losses import mean_squared_error

import argparse
# 模型参数解析设置
# For HPD model, we use the YPhy = 1 and lamda (i.e., \lambda_{phy}) = 0

# 模型预训练阶段
ap = argparse.ArgumentParser(description='pretraining')
# 模型数据集选择
ap.add_argument('--Boost_Dataset', choices=['mille_lacs','mendota'], default='mille_lacs', type=str, help='Dataset choice')
# 模型优化器选择
ap.add_argument('--optimizer_val', choices=['0: Adagrad', '1: Adadelta', '2: Adam', '3: Nadam', '4: RMSprop', '5: SGD', '6: NSGD'], default=2, type=int, help='Optimizer')
# 数据所在文件夹路径
ap.add_argument('--data_dir', default='../datasets/', type=str, help='Data Directory')
# 模型单次训练所使用的数据批次大小（控制一次训练中样本数量，影响显存占用与梯度估计稳定性）
ap.add_argument('--batch_size', default=1000, type=int, help='Batch Size')
# 最大训练轮数（指定最大训练时间 实际中早停（patience_val）会提前中止）
ap.add_argument('--epochs', default=10000, type=int, help='Epochs')
# Dropout比例，用于正则化防止过拟合（设置为 0.1~0.3 可以帮助小数据训练时提高泛化能力）
ap.add_argument('--drop_frac', default=0.0, type=float, help='Dropout Fraction')
# 是否将物理模型输出（如仿真/经验公式得到的值）加入到神经网络的输入中）
ap.add_argument('--use_YPhy', type=int, default=1, help='Use Physics Numeric Model as input')
# 模型维度（神经元数量 单位隐藏层包含12个神经元）
ap.add_argument('--n_nodes', default=12, type=int, help='Number of Nodes in Hidden Layer')
# 模型深度（隐藏层数量 <3层 浅层神经网络 传统机器学习 并非深度学习）
ap.add_argument('--n_layers', default=2, type=int, help='Number of Hidden Layers')
# 模型超参数
ap.add_argument('--lamda', default=0.0, type=float, help='lambda hyperparameter')
# 模型训练数据集规模
ap.add_argument('--tr_size', default=3000, type=int, help='Size of Training set')
# 模型验证分数（验证数据集的比例 10% 数据集）
ap.add_argument('--val_frac', default=0.1, type=float, help='Validation Fraction')
# 模型提前停止（EarlyStopping）机制的耐心值（patience）如果在连续500个epoch中验证集损失（val_loss）没有提升 就提前终止训练 防止过拟合
ap.add_argument('--patience_val', default=500, type=int, help='Patience Value for Early Stopping')
# 实验随机运行次数（随机种子初始化 + 重新训练）
ap.add_argument('--n_iters', default=1, type=int, help='Number of Random Runs')
#  数据结果保存路径
ap.add_argument('--save_dir', default='./results/', type=str, help='Save Directory')
args = ap.parse_args()

#function to compute the room_mean_squared_error given the ground truth (y_true) and the predictions(y_pred)
# 经验损失：均方误差（RMSE）
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

#function for computing the density given the temperature(nx1 matrix)
# 密度计算函数（原始物理模型）
def density(temp):
    return 1000 * ( 1 - (temp + 288.9414) * (temp - 3.9863)**2 / (508929.2 * (temp + 68.12963) ) )

# 物理损失函数
def phy_loss_mean(params):
	# useful for cross-checking training
    udendiff, lam = params
    def loss(y_true,y_pred):
        return K.mean(K.relu(udendiff))
    return loss

#function to calculate the combined loss = sum of rmse and phy based loss
# 联合损失函数：经验损失 + 物理损失
def combined_loss(params):
    udendiff, lam = params
    def loss(y_true,y_pred):
        return mean_squared_error(y_true, y_pred) + lam * K.mean(K.relu(udendiff))
    return loss

# 激活函数 ReLU 的 Numpy 实现
def relu(m):
    m[m < 0] = 0
    return m

# 离线评估物理一致性
def evaluate_physics_loss(model, uX1, uX2):
    tolerance = 0
    uout1 = model.predict(uX1)
    uout2 = model.predict(uX2)
    udendiff = (density(uout1) - density(uout2))
    percentage_phy_incon = np.sum(udendiff>tolerance)/udendiff.shape[0]
    phy_loss = np.mean(relu(udendiff))
    return phy_loss, percentage_phy_incon

# 物理引导神经网络训练函数
def PGNN_train_test(iteration=0):
    # Hyper-parameters of the training process
    # 模型训练超参数
    batch_size = args.batch_size      # 单批次训练规模
    num_epochs = args.epochs          # 训练批次
    val_frac = args.val_frac          # 验证比例
    patience_val = args.patience_val  # 耐心值防止过拟合
    
    # 可选优化器 List of optimizers to choose from
    optimizer_names = ['Adagrad', 'Adadelta', 'Adam', 'Nadam', 'RMSprop', 'SGD', 'NSGD']
    optimizer_vals = [Adagrad(clipnorm=1), Adadelta(clipnorm=1), Adam(clipnorm=1), Nadam(clipnorm=1), RMSprop(clipnorm=1), SGD(clipnorm=1.), SGD(clipnorm=1, nesterov=True)]
    
    # 选择优化器 selecting the optimizer
    optimizer_name = optimizer_names[args.optimizer_val]
    optimizer_val = optimizer_vals[args.optimizer_val]

    # 数据集加载
    data_dir = args.data_dir
    filename = args.dataset + '.mat'
    mat = spio.loadmat(data_dir + filename, squeeze_me=True,
    variable_names=['Y','Xc_doy','Modeled_temp'])
    Xc = mat['Xc_doy']           # 输入特征
    Y = mat['Y']                 # 监督标签
    YPhy = mat['Modeled_temp']   # 物理模型预测值
    trainX, trainY = Xc[:args.tr_size,:],Y[:args.tr_size]
    testX, testY = Xc[args.tr_size:,:],Y[args.tr_size:]

    # Loading unsupervised data
    # 无监督物理样本加载
    unsup_filename = args.dataset + '_sampled.mat'
    unsup_mat = spio.loadmat(data_dir+unsup_filename, squeeze_me=True,
    variable_names=['Xc_doy1','Xc_doy2'])
    
    uX1 = unsup_mat['Xc_doy1'] # Xc at depth i for every pair of consecutive depth values
    uX2 = unsup_mat['Xc_doy2'] # Xc at depth i + 1 for every pair of consecutive depth values

    if args.use_YPhy == 0:
    	# Removing the last column from uX (corresponding to Y_PHY)
        uX1 = uX1[:,:-1]
        uX2 = uX2[:,:-1]
        trainX = trainX[:,:-1]
        testX = testX[:,:-1]
    
    # Creating the model
    model = Sequential()
    for layer in np.arange(args.n_layers):
        if layer == 0:
            model.add(Dense(args.n_nodes, activation='relu', input_shape=(np.shape(trainX)[1],)))
        else:
             model.add(Dense(args.n_nodes, activation='relu'))
        model.add(Dropout(args.drop_frac))
    model.add(Dense(1, activation='linear'))
    
    # physics-based regularization
    uin1 = K.constant(value=uX1) # input at depth i
    uin2 = K.constant(value=uX2) # input at depth i + 1
    lam = K.constant(value=args.lamda) # regularization hyper-parameter
    uout1 = model(uin1) # model output at depth i
    uout2 = model(uin2) # model output at depth i + 1
    udendiff = (density(uout1) - density(uout2)) # difference in density estimates at every pair of depth values
    
    totloss = combined_loss([udendiff, lam])
    phyloss = phy_loss_mean([udendiff, lam])

    model.compile(loss=totloss,
                  optimizer=optimizer_val,
                  metrics=[phyloss, root_mean_squared_error])

    early_stopping = EarlyStopping(monitor='val_loss_1', patience=args.patience_val, verbose=1)
    
    print('Running...' + optimizer_name)
    history = model.fit(trainX, trainY,
                        batch_size = args.batch_size,
                        epochs = args.epochs,
                        verbose = 0,
                        validation_split = args.val_frac, callbacks=[early_stopping, TerminateOnNaN()])
    
    test_score = model.evaluate(testX, testY, verbose=0)

    phy_cons, percent_phy_incon = evaluate_physics_loss(model, uX1, uX2)
    test_rmse = test_score[2]
    train_rmse = history.history['root_mean_squared_error'][-1]
    
    print(" Train RMSE = ", train_rmse)
    print(" Test RMSE = ", test_rmse)
    print(" Physical Consistency = ", phy_cons)
    print(" Percentage Physical Incon = ", percent_phy_incon)
    
    exp_name = 'pgnn_'+args.dataset+optimizer_name + '_drop' + str(args.drop_frac) + '_usePhy' + str(args.use_YPhy) +  '_nL' + str(args.n_layers) + '_nN' + str(args.n_nodes) + '_trsize' + str(args.tr_size) + '_lamda' + str(args.lamda) + '_iter' + str(iteration)
    exp_name = exp_name.replace('.','pt')
    results_name = args.save_dir + exp_name + '_results.mat' # storing the results of the model
    spio.savemat(results_name, 
                 {'train_loss_1':history.history['loss_1'], 
                  'val_loss_1':history.history['val_loss_1'], 
                  'train_rmse':history.history['root_mean_squared_error'], 
                  'val_rmse':history.history['val_root_mean_squared_error'], 
                  'test_rmse':test_score[2]})
    
    return train_rmse, test_rmse, phy_cons, percent_phy_incon



for iteration in range(args.n_iters):
    train_rmse, test_rmse, phy_cons, percent_phy_incon = PGNN_train_test(iteration)
