'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
from tensorflow.keras.layers import Conv1D, Multiply, Activation, LSTM, Dense
tf.disable_v2_behavior()
import os
import numpy as np
from tqdm import tqdm

from utils import normalization, renormalization, rounding
from utils import xavier_init
from utils import binary_sampler, uniform_sampler, sample_batch_index, sliding_window_sampling, sliding_window_sampling2
from tensorflow.keras.optimizers.schedules import ExponentialDecay  # 引入 ExponentialDecay
import matplotlib.pyplot as plt

# 检查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# import torch
# import torch.nn as nn

# Split the data into training and test sets
def train_test_split(data, test_ratio=0.2):
    num_samples = data.shape[0]
    num_test_samples = int(num_samples * test_ratio)
    train_data = data[:num_samples - num_test_samples]
    test_data = data[num_samples - num_test_samples:]
    return train_data, test_data

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, feature_dim, max_len=64):
        super(PositionalEncoding, self).__init__()
        pe = np.zeros((max_len, feature_dim))
        position = np.expand_dims(np.arange(0, max_len), 1)
        div_term = np.exp(np.arange(0, feature_dim, 2) * -(np.log(10000.0) / feature_dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = tf.constant(pe, dtype=tf.float32)
        
    def call(self, x):
        sequence_length = tf.shape(x)[0]
        return x + self.pe[:sequence_length, :]


    
class CustomGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super(CustomGRU, self).__init__()
        self.units = units
        self.Wr = Dense(units)  # 重置门
        self.Ur = Dense(units)
        self.Wz = Dense(units)  # 更新门
        self.Uz = Dense(units)
        self.Wh = Dense(units)  # 候选隐藏状态
        self.Uh = Dense(units)

    def call(self, x, h_prev):
        print("shape of x and h_prev: ", x.shape, h_prev.shape)
        # 重置门 r_t
        r_t = tf.sigmoid(self.Wr(x) + self.Ur(h_prev))
        # 更新门 z_t
        z_t = tf.sigmoid(self.Wz(x) + self.Uz(h_prev))
        # 候选隐藏状态 h_t~
        h_t_candidate = tf.tanh(self.Wh(x) + self.Uh(r_t * h_prev))
        # 最终隐藏状态 h_t
        h_t = z_t * h_prev + (1 - z_t) * h_t_candidate
        hidden_state = tf.expand_dims(h_t, -1)
        print("shape of h_t and hidden_state: ", h_t.shape, hidden_state.shape)
        return h_t, hidden_state



class GCNGRU(tf.keras.layers.Layer):
    def __init__(self, batch_size, num_output_channels, num_motif, order, num_columns_of_gru):
        super(GCNGRU, self).__init__()
        self.batch_size = batch_size
        self.num_output_channels = num_output_channels
        self.num_motif = num_motif
        self.order = order
        self.num_columns_of_gru = num_columns_of_gru
        self.num_layers = self.batch_size 

        self.gcn_weights_biases_x = []  # 用于存储每一层的 gcn 权重和偏置
        self.gcn_weights_biases_h = []  

        self.gru_layers = []

        for i in range(self.num_layers):
            if i == 0:
                # 第一层使用 [1, 2, num_motif * order + order + 1] 形状的权重
                gcn_weights_h = tf.Variable(xavier_init([1, 2, num_motif * order + order + 1]))#隐藏层处理的参数
            else:
                #第二层及以后的层使用 [1, 1, num_motif * order + order + 1] 形状的权重
                gcn_weights_h = tf.Variable(xavier_init([1, 1, num_motif * order + order + 1]))#输入数据处理的参数

            gcn_weights_x = tf.Variable(xavier_init([1, 2, num_motif * order + order + 1]))

            gcn_bias_x = tf.Variable(tf.zeros([1, 16])) #时间步数，节点数
            gcn_bias_h = tf.Variable(tf.zeros([1, 16]))

            self.gcn_weights_biases_x.append((gcn_weights_x, gcn_bias_x))
            self.gcn_weights_biases_h.append((gcn_weights_h, gcn_bias_h))

            self.gru_layers.append(CustomGRU(units=16))
            
    def call(self, inputs, a_list):
        batch_size, num_nodes, feature_dim = inputs.shape
        hidden_state = tf.zeros([1, num_nodes, feature_dim]) # 初始化隐藏状态
        output_sequence = []

        for t in range(self.num_layers):
            # 提取当前时间步的数据
            x_t = inputs[t, :, :]
            x_t = tf.reshape(x_t, [1, num_nodes, feature_dim])  # 批量大小，节点数, 特征数
            # 使用 GCN 更新输入特征 x_t 和隐藏状态
            gcn_weights_x, gcn_bias_x = self.gcn_weights_biases_x[t % self.num_layers]
            gcn_weights_h, gcn_bias_h = self.gcn_weights_biases_h[t % self.num_layers]

            x_t_gcn = gcn_layer(x_t, self.num_output_channels, a_list, gcn_weights_x, gcn_bias_x, self.order, self.num_motif)

            if t != 0:
                hidden_state_gcn = gcn_layer(hidden_state, self.num_output_channels, a_list, gcn_weights_h, gcn_bias_h, self.order, self.num_motif)
            else:
                hidden_state_gcn = tf.zeros([1, num_nodes])  # 形状变为 [1, 16]

            # 使用自定义 GRU 更新隐藏状态
            result, hidden_state = self.gru_layers[t % self.num_layers](x_t_gcn, hidden_state_gcn)
            output_sequence.append(result)

        output_sequence = tf.concat(output_sequence, axis=0)
        print("shape of output_sequence: ", output_sequence.shape)
        return output_sequence


    

# GCN utilities
def gcn_layer(X, num_output_channels, A_list, weight, bias, order, num_motif):
    """Graph Convolutional Network layer.
    Args:
        X: Input feature matrix.
        A: Adjacency matrix.
        weight: Weight matrix for GCN layer.
        bias: Bias vector for GCN layer.
    Returns:
        Output feature matrix after applying GCN layer.
    """
    L_hat_list = []
    for A in A_list:
        D = tf.linalg.tensor_diag(tf.reduce_sum(A, axis=0))
        L = D - A
        D_inv_sqrt = tf.linalg.inv(tf.sqrt(D))
        ##########################################
        L_hat = D_inv_sqrt @ L @ D_inv_sqrt # Normalized Laplacian in Eq. (20)
        L_hat_list.append(L_hat)
    

    # Number of output channels
    num_input_channels = X.shape[-1]
    num_output_channels = num_output_channels
    outputs = []

    print("shape of X: ", X.shape)
    # Polynomial filter as in Eq. (21)
    for i in range(num_output_channels):
        P_output_list = []

        for j in range(num_input_channels):
            
            P = [tf.eye(int(L_hat_list[0].shape[0]))]# 包含多个m*m的张量，第一个张量为单位阵
            # 加权和
            weighted_sum = weight[i, j, num_motif*order+1] * P[0]#先把p[0]加入，-1表示最后一个元素

            for k in range(1, order): # 多项式设计
                P_next = tf.zeros_like(P[0])
                
                for l in range(num_motif):
                    P_next +=weight[i,j,l*(k-1)] * tf.matmul(L_hat_list[l], P[-1])
                P.append(P_next)
                weighted_sum += weight[i, j, num_motif*order+k+1] * P[-1]
            #########################################
            P_output_list.append(weighted_sum)  # Accumulate polynomial filters

        P_output = tf.stack(P_output_list, axis=0)

        #print("shape of P_output: ", P_output.shape)

        # 对每个通道分别与 P_output 相乘，然后加和
        P_outputs = [tf.matmul(P_output[j], tf.transpose(X[:,:,j])) for j in range(num_input_channels)]
        #print("shape of P_output[j]: ", P_output[j].shape)
        #for n, P_output_elem in enumerate(P_outputs):
            #print(f"shape of P_outputs[{n}]: ", P_output_elem.shape)
        
        # 加和所有通道的结果
    
        P_output_sum =tf.nn.relu(tf.add_n(P_outputs) + tf.transpose(bias))
        #print("shape of P_output_sum: ", P_output_sum.shape)
        outputs.append(P_output_sum)
    print("shape of gcn_output: ", outputs[-1].shape)
    
    return tf.transpose(outputs[-1]) #这里偷懒了，假设了输出通道只有一个元素

def gain (data_x, full_data_x, gain_parameters, A_list):
    # Define mask matrix
    
    data_m = 1-np.isnan(data_x)
    
      # Split data into training and test sets
    train_x, test_x = train_test_split(data_x)
    train_m, test_m = train_test_split(data_m)
    train_x_full, test_x_full = train_test_split(full_data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    num_motif = gain_parameters['num_motif']
    order = gain_parameters['order']
    step_size = gain_parameters['step_size']
    filters = gain_parameters['filters']
    kernel_size = gain_parameters['kernel_size']
    learning_rate = gain_parameters['learning_rate']
    num_columns_of_gru = gain_parameters['num_columns_of_gru']

    # 全局步数定义
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # 学习率调度器的定义
    learning_rate_schedule = ExponentialDecay(
    initial_learning_rate=learning_rate,  # 初始学习率
    decay_steps=1000,  # 每隔 1000 个训练步骤进行一次学习率衰减
    decay_rate=0.96,  # 每次衰减的比例
    staircase=True  # 如果设置为 True 则学习率是阶梯下降，否则是连续下降
    )

    # 获取学习率的当前值
    current_learning_rate = learning_rate_schedule(global_step)

    
    no, dim = data_x.shape
    
    
    # Normalization
    norm_train_data, norm_parameters = normalization(train_x)
    norm_test_data, _ = normalization(test_x, norm_parameters)
    norm_train_x = np.nan_to_num(norm_train_data, 0)
    norm_test_x = np.nan_to_num(norm_test_data, 0)

    
    # 创建 PositionalEncoding 层
    positional_encoding_layer = PositionalEncoding(feature_dim=dim, max_len=batch_size)

    
    
    ## GAIN architecture   
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape = [batch_size, dim]) # 数据转置了，所以行是特征数，列是样本数
    # Mask vector 
    M = tf.placeholder(tf.float32, shape = [batch_size, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape = [batch_size, dim])
    A_list_ph = [tf.placeholder(tf.float32, shape=[dim, dim]) for _ in range(num_motif)]

    #Generator variables
    G_W1 = tf.Variable(xavier_init([dim, dim]))
    G_b1 = tf.Variable(tf.zeros([batch_size, dim]))
    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim, dim]))
    D_b1 = tf.Variable(tf.zeros([batch_size, dim]))

    ## GAIN functions
    # Generator
    def generator(x, m, a_list):
        # 应用正余弦编码
        x_encoded = positional_encoding_layer(x)
        m_encoded = positional_encoding_layer(m)
    
        # 创建并应用 GCNGRU 层
        gcn_gru_layer = GCNGRU(batch_size=batch_size, num_output_channels=1, num_motif=num_motif, order=order, num_columns_of_gru = num_columns_of_gru)
        combined_inputs = tf.stack([x_encoded, m_encoded], axis=-1)
        G_h1 = gcn_gru_layer(combined_inputs, a_list)
        G_logit = tf.matmul(G_h1, G_W1) + G_b1#全连接层
        G_prob = tf.nn.sigmoid(G_logit)
        return  G_prob

    # Discriminator
    def discriminator(x, h, a_list):
        # 应用正余弦编码
        x_encoded = positional_encoding_layer(x)
        h_encoded = positional_encoding_layer(h)
    
        # 创建并应用 GCNGRU 层
        gcn_gru_layer = GCNGRU(batch_size=batch_size, num_output_channels=1, num_motif=num_motif, order=order, num_columns_of_gru = num_columns_of_gru)
        combined_inputs = tf.stack([x_encoded, h_encoded], axis=-1)
        D_h1 = gcn_gru_layer(combined_inputs, a_list)
        D_logit = tf.matmul(D_h1, D_W1) + D_b1
        D_prob = tf.nn.sigmoid(D_logit)#全连接层
        return D_prob

  
    ## GAIN structure
    # Generator
    with tf.variable_scope("generator"):
        G_sample = generator(X, M, A_list_ph)
 
    # Combine with observed data
    Hat_X = X * M + G_sample * (1-M)
  
    # Discriminator
    with tf.variable_scope("discriminator"):
        D_prob = discriminator(Hat_X, H, A_list_ph)
  
    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                  + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
    G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
    MSE_loss = \
    tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss 
    
    # 在定义 D_solver 之前检查变量
    #for var in tf.trainable_variables():
    #    print(var.name, tf.gradients(D_loss, var))

    # 获取所有可训练变量，包括显式和隐式的变量
    trainable_vars = tf.trainable_variables()

    # 使用作用域名称来区分生成器和判别器的变量
    theta_G = [var for var in trainable_vars if var.name.startswith("generator")]
    theta_D = [var for var in trainable_vars if var.name.startswith("discriminator")]

    # 计算判别器的梯度
    D_gradients = tf.gradients(D_loss, theta_D)



    # 对梯度进行裁剪，限制在 [-1, 1] 范围内
    #D_gradients_clipped = [
    #    tf.clip_by_value(grad, -20.0, 20.0) if grad is not None else None
    #    for grad in D_gradients
    #]
    # 对梯度进行范数裁剪，限制范数最大为 10
    D_gradients_clipped, _ = tf.clip_by_global_norm(D_gradients, 20.0)
    # 使用裁剪后的梯度来更新判别器参数，保持全局步数更新
    D_solver = tf.train.AdamOptimizer(learning_rate=current_learning_rate).apply_gradients(
        [(grad, var) for grad, var in zip(D_gradients_clipped, theta_D) if grad is not None]
    )

    # 计算生成器的梯度
    G_gradients = tf.gradients(G_loss, theta_G)
    # 对梯度进行裁剪，限制在 [-1, 1] 范围内
    #G_gradients_clipped = [
    #    tf.clip_by_value(grad, -20.0, 20.0) if grad is not None else None
    #    for grad in G_gradients
    #]
    # 对梯度进行范数裁剪，限制范数最大为 10
    G_gradients_clipped, _ = tf.clip_by_global_norm(G_gradients, 20.0)
    #使用裁剪后的梯度来更新生成器参数，保持全局步数更新
    G_solver = tf.train.AdamOptimizer(learning_rate=current_learning_rate).apply_gradients(
        [(grad, var) for grad, var in zip(G_gradients_clipped, theta_G) if grad is not None]
    )

  
    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
  
    # 生成批次
    train_batches_x = sliding_window_sampling2(norm_train_x, batch_size, step_size)
    train_batches_m = sliding_window_sampling2(train_m, batch_size, step_size)
    num_batches = len(train_batches_x)
    
    # 保存每次迭代的损失
    d_losses = []
    g_losses = []
    mse_losses = []

    # 开始迭代
    batch_idx = 0



    # Start Iterations
    for it in tqdm(range(iterations)):    

        if batch_idx >= num_batches:
            batch_idx = 0  # 重新开始新的epoch
            print(f"Iteration: {it}, Current discriminator loss: {D_loss_curr}")
            print(f"Iteration: {it}, Current Generator loss: {G_loss_curr + alpha * MSE_loss_curr}")
            print(f"Iteration: {it}, Current MSE loss: {MSE_loss_curr}")
        

        # Sample batch
        X_mb = train_batches_x[batch_idx]
        M_mb = train_batches_m[batch_idx]
        Z_mb = uniform_sampler(0, 0.01, batch_size, train_x.shape[1])
        H_mb_temp = binary_sampler(hint_rate, batch_size, train_x.shape[1])
        H_mb = M_mb * H_mb_temp

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        feed_dict = {X: X_mb, M: M_mb, H: H_mb}
        feed_dict.update({a_ph: a_mb for a_ph, a_mb in zip(A_list_ph, A_list)})#不能直接传列表A_list，需要更新

        _, D_loss_curr = sess.run([D_solver, D_loss_temp], 
                                  feed_dict = feed_dict)
        _, G_loss_curr, MSE_loss_curr = \
        sess.run([G_solver, G_loss_temp, MSE_loss],
                 feed_dict = feed_dict)


        # 保存每次迭代的损失
        d_losses.append(D_loss_curr)
        g_losses.append(G_loss_curr + alpha * MSE_loss_curr)
        mse_losses.append(MSE_loss_curr)

        # 更新批次索引
        batch_idx += 1
    # 创建保存模型的 Saver 对象
    saver = tf.train.Saver()
    # 定义模型保存路径
    save_path = './models/gain10_model10'
    # 检查点目录
    if not os.path.exists('./models'):
        os.makedirs('./models')
    # Other parameters
    # 保存训练结束后的模型
    saver.save(sess, save_path)
    #model.save(save_path)
    print(f"Model saved in path: {save_path}")

    # 绘制损失图并保存
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), d_losses, label='Discriminator Loss', color='b')
    plt.plot(range(iterations), g_losses, label='Generator Loss', color='r')
    plt.plot(range(iterations), mse_losses, label='MSE Loss', color='g')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Losses during GAIN Training')
    plt.savefig('data/gain_loss_plot.png')
    plt.show()

    #插补阶段也分批次
    # Imputation phase for both training and test sets
    def impute_data(norm_data, data_m):
        imputed_data = np.zeros_like(norm_data)
        for batch_start in range(0, norm_data.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, norm_data.shape[0])
            batch_slice = slice(batch_start, batch_end)

            X_mb = norm_data[batch_slice, :]
            M_mb = data_m[batch_slice, :]
            Z_mb = uniform_sampler(0, 0.01, batch_size, norm_data.shape[1])
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            imputed_batch = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb, **{a_ph: a_mb for a_ph, a_mb in zip(A_list_ph, A_list)}})[0]
            imputed_data[batch_slice, :] = imputed_batch
        return imputed_data
    
    imputed_train_data = impute_data(norm_train_x, train_m)
    imputed_test_data = impute_data(norm_test_x, test_m)

    #imputed_train_data = train_m * norm_train_x + (1 - train_m) * imputed_train_data
    #imputed_test_data = test_m * norm_test_x + (1 - test_m) * imputed_test_data
  
    # Renormalization
    imputed_train_data = renormalization(imputed_train_data, norm_parameters)
    imputed_test_data = renormalization(imputed_test_data, norm_parameters)

    # Rounding
    imputed_train_data = rounding(imputed_train_data, train_x)
    imputed_test_data = rounding(imputed_test_data, test_x)

    # Calculate test loss
    test_loss = np.mean(((1 - test_m) * test_x_full - (1 - test_m) * imputed_test_data) ** 2) / np.mean(test_m)
    print(f"Test Loss: {test_loss}")

    return imputed_train_data, imputed_test_data
