import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
from tcn.tcn import TemporalConvNet
import numpy as np
print('Net')
def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

# def gram_loss(xn1,xn2,mu1,mu2):
#
def lstm_cell(lstm_size, keep_prob=1.0):
  lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
  drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
  return drop

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def TCN_G(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    linear = tf.contrib.layers.fully_connected(tcn, output_size,activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.01))
    return linear

def TCN_G_roll(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout,pre_length):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    print('tcntcn')
    print(tcn)
    print(tcn.shape[1]*tcn.shape[2])
    tcn=tf.reshape(tcn,[tcn.shape[0],tcn.shape[1]*tcn.shape[2]])
    tcn = tf.contrib.layers.fully_connected(tcn, int(pre_length * output_size), activation_fn=tf.nn.tanh,
                                            weights_initializer=tf.random_normal_initializer(0, 0.01))
    print('tcn')
    linear = tf.reshape(tcn,[tcn.shape[0],pre_length,output_size])
    return linear

def TCN_D(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    print(tcn)
    tcn=tf.reshape(tcn,[tcn.shape[0],tcn.shape[1]*tcn.shape[2]])
    linear = tf.contrib.layers.fully_connected(tcn, output_size,activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.01))
    print('zhenghaoran',linear)
    return linear

def TCN_C(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    linear = tf.contrib.layers.fully_connected(tcn, output_size,activation_fn=tf.nn.tanh, weights_initializer=tf.random_normal_initializer(0, 0.01))
    return linear
def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
  """ 2D normal distribution
  input
  - x,mu: input vectors
  - s1,s2: standard deviances over x1 and x2
  - rho: correlation coefficient in x1-x2 plane
  """
  # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
  norm1 = tf.subtract(x1, mu1)
  norm2 = tf.subtract(x2, mu2)
  s1s2 = tf.multiply(s1, s2)
  z = tf.square(tf.div(norm1, s1))+tf.square(tf.div(norm2, s2))-2.0*tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
  negRho = 1-tf.square(rho)
  result = tf.exp(tf.div(-1.0*z,2.0*negRho))
  denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
  px1x2 = tf.div(result, denom)
  return px1x2
#TCN+MDN

decay=[]
step=100
for i in range(step):
    d=0.95**(i/step)
    decay.append(d)
decay=np.array(decay).T
# print(decay)

def TCN_MDN(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout,mixtures,labels,pad):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length,
                          kernel_size=kernel_size, dropout=dropout,use_highway=False)
    # print(tcn)
    # print('fffffff')
    batch_size=input_layer.shape[0]
    linear = tf.contrib.layers.fully_connected(tcn, output_size, activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01))
    linear=tf.transpose(linear,[0,2,1])
    with tf.name_scope("Output_MDN") as scope:
        params =6  # 7+theta
        # Two for distribution over hit&miss, params for distribution parameters
        output_units = mixtures * params
        W_o = tf.Variable(tf.random_normal(
            [output_size, output_units], stddev=0.01))
        print('ddddddd')
        b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))
        print(b_o)
        # For comparison with XYZ, only up to last time_step
        # --> because for final time_step you cannot make a prediction
        # print('lllll')
        # print(linear)
        outputs = tf.transpose(linear, [2,0,1])
        print('out', outputs.shape)
        # outputs_tensor=tf.reshape(outputs,[])
        # output = outputs
        # print('out', output.shape)
        # # outputs_tensor = tf.concat(output, axis=0)
        outputs_tensor = tf.reshape(outputs, [outputs.shape[0] * outputs.shape[1], outputs.shape[2]])
        # print(outputs_tensor.shape)
        # is of size [batch_size*seq_len by output_units]
        h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)

    with tf.name_scope('MDN_over_next_vector') as scope:
        # Next two lines are rather ugly, But its the most efficient way to
        # reshape the data
        print('hhhh')
        print(h_out_tensor)
        h_xyz = tf.reshape(h_out_tensor, (sequence_length, batch_size, output_units))
        # transpose to [batch_size, output_units, sl-1]
        h_xyz = tf.transpose(h_xyz, [1, 2, 0])
        # x_next = tf.slice(x,[0,0,1],[batch_size,3,sl-1])  #in size [batch_size,
        # output_units, sl-1]
        # x_next = tf.subtract(self.x[:, :2, 1:], self.x[:, :2, :sl - 1])
        # From here any, many variables have size [batch_size, mixtures, sl-1]
        # xn1, xn2 = tf.split(value=x_next, num_or_size_splits=2, axis=1)
        xn1=labels[:,:,0]
        xn2=labels[:,:,1]
        print('liyangggg')
        print(xn1)
        print(labels)
        xn1=tf.reshape(xn1,[xn1.shape[0],xn1.shape[1],1])
        xn1=tf.transpose(xn1,[0,2,1])
        print('shape')
        print(xn1)
        xn2 = tf.reshape(xn2, [xn2.shape[0], xn2.shape[1],1])
        xn2 = tf.transpose(xn2, [0, 2, 1])
        print('h_xyz')
        print(h_xyz)
        mu1, mu2, s1, s2,  rho, theta = tf.split(value=h_xyz, num_or_size_splits=params, axis=1)
        print('mu1')
        print(s1)

        # m   ke the theta mixtures
        # softmax all the theta's:
        max_theta = tf.reduce_max(theta, 1, keep_dims=True)
        theta = tf.subtract(theta, max_theta)
        theta = tf.exp(theta)
        normalize_theta = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
        theta = tf.multiply(normalize_theta, theta)

        # Deviances are non-negative and tho between -1 and 1
        s1 = tf.exp(s1)
        s2 = tf.exp(s2)
        # self.s3 = tf.exp(self.s3)
        rho = tf.tanh(rho)

        # probability in x1x2 plane
        px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2,s1, s2, rho)
        # px3 = tf_1d_normal(xn3, self.mu3, self.s3)
        # px1x2x3 = tf.multiply(px1x2, px3)

        # Sum along the mixtures in dimension 1
        px1x2x3_mixed = tf.reduce_sum(tf.multiply(px1x2, theta), 1)
        print('You are using %.0f mixtures' % mixtures)
        # at the beginning, some errors are exactly zero.
        loss_seq = -tf.log(0.1*tf.maximum(px1x2x3_mixed, 1e-20))
        print(loss_seq)
        loss_seq = decay * loss_seq
        # print(loss_seq)
        # print(pad)
        print('gggg')
        pad0=tf.reshape(pad,[pad.shape[0],pad.shape[1]])
        loss_seq=pad0*loss_seq
        print(loss_seq)
        num = tf.count_nonzero(pad0,axis=1)
        sum=tf.reduce_sum(num)

        sum = tf.cast(sum,tf.float32)

        # print(loss_seq)
        # loss_seq=loss_seq[:,:num]
        # print(loss_seq.shape)
        cost_seq = tf.reduce_sum(loss_seq)/sum


    return cost_seq,mu1,mu2,s1,s2,rho,theta
###############################################  mlp #############################################
#20201219新加


class G_TCN_MDN(object):
#用来产生概率分布
    def __init__(self):
        self.name='G_TCN_MDN'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout,labels,pad):
        with tf.variable_scope(self.name) as vs:
            cost, mu1, mu2, s1, s2, rho, theta = TCN_MDN(z, 10, channel_sizes, seq_len, kernel_size=kernel_size,
                                                         dropout=dropout, mixtures=1, labels=labels,pad=pad)
            return cost, mu1, mu2, s1, s2, rho, theta
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


#滚动预测的代码
def TCN_MDN_roll_abanden(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout,mixtures,labels,pre_length):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length,
                          kernel_size=kernel_size, dropout=dropout,use_highway=False)
    print(tcn)
    print('fffffff')
    tem=tcn.shape[2]
    tcn=tf.reshape(tcn,[tcn.shape[0],tcn.shape[1]*tcn.shape[2]])
    tcn=tf.contrib.layers.fully_connected(tcn, int(pre_length*output_size), activation_fn=tf.nn.relu6,weights_initializer=tf.random_normal_initializer(0, 0.01))
    print('tcn')
    print(tcn)
    tcn=tf.reshape(tcn,[tcn.shape[0],pre_length,output_size])
    batch_size=input_layer.shape[0]
    # linear = tf.contrib.layers.fully_connected(tcn, output_size, activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01))
    linear=tcn
    print(linear)
    linear=tf.transpose(linear,[0,2,1])
    with tf.name_scope("Output_MDN") as scope:
        params =6  # 7+theta
        # Two for distribution over hit&miss, params for distribution parameters
        output_units = mixtures * params
        W_o = tf.Variable(tf.random_normal(
            [output_size, output_units], stddev=0.01))
        print('ddddddd')
        b_o = tf.Variable(tf.constant(0.5, shape=[output_units]))
        print(b_o)
        # For comparison with XYZ, only up to last time_step
        # --> because for final time_step you cannot make a prediction
        # print('lllll')
        # print(linear)
        outputs = tf.transpose(linear, [2,0,1])
        print('out', outputs.shape)
        # outputs_tensor=tf.reshape(outputs,[])
        # output = outputs
        # print('out', output.shape)
        # # outputs_tensor = tf.concat(output, axis=0)
        outputs_tensor = tf.reshape(outputs, [outputs.shape[0] * outputs.shape[1], outputs.shape[2]])
        # print(outputs_tensor.shape)
        # is of size [batch_size*seq_len by output_units]
        h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)

    with tf.name_scope('MDN_over_next_vector') as scope:
        # Next two lines are rather ugly, But its the most efficient way to
        # reshape the data
        print('hhhh')
        print(h_out_tensor)
        h_xyz = tf.reshape(h_out_tensor, (pre_length, batch_size, output_units))
        # transpose to [batch_size, output_units, sl-1]
        h_xyz = tf.transpose(h_xyz, [1, 2, 0])
        # x_next = tf.slice(x,[0,0,1],[batch_size,3,sl-1])  #in size [batch_size,
        # output_units, sl-1]
        # x_next = tf.subtract(self.x[:, :2, 1:], self.x[:, :2, :sl - 1])
        # From here any, many variables have size [batch_size, mixtures, sl-1]
        # xn1, xn2 = tf.split(value=x_next, num_or_size_splits=2, axis=1)
        xn1=labels[:,:,0]
        xn2=labels[:,:,1]
        # a_v=labels[:,:,2]
        # a_a=labels[:,:,3]
        # a_v=tf.reshape(a_v,[a_v.shape[0],1,a_v.shape[1]])
        # a_a=tf.reshape(a_a,[a_a.shape[0],1,a_a.shape[1]])
        print('liyangggg')
        print(xn1)
        print(labels)
        xn1=tf.reshape(xn1,[xn1.shape[0],xn1.shape[1],1])
        xn1=tf.transpose(xn1,[0,2,1])
        print('shape')
        print(xn1)
        xn2 = tf.reshape(xn2, [xn2.shape[0], xn2.shape[1],1])
        xn2 = tf.transpose(xn2, [0, 2, 1])
        print('h_xyz')
        print(h_xyz)
        # mu1, mu2, s1, s2,  rho, theta,v,a = tf.split(value=h_xyz, num_or_size_splits=params, axis=1)
        mu1, mu2, s1, s2,  rho, theta = tf.split(value=h_xyz, num_or_size_splits=params, axis=1)
        print('mu1')
        print(s1)

        # m   ke the theta mixtures
        # softmax all the theta's:
        max_theta = tf.reduce_max(theta, 1, keep_dims=True)
        theta = tf.subtract(theta, max_theta)
        theta = tf.exp(theta)
        normalize_theta = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
        theta = tf.multiply(normalize_theta, theta)

        # Deviances are non-negative and tho between -1 and 1
        s1 = tf.exp(s1)
        s2 = tf.exp(s2)
        # self.s3 = tf.exp(self.s3)
        rho = tf.tanh(rho)

        # probability in x1x2 plane
        px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2,s1, s2, rho)
        # px3 = tf_1d_normal(xn3, self.mu3, self.s3)
        # px1x2x3 = tf.multiply(px1x2, px3)

        # Sum along the mixtures in dimension 1
        px1x2x3_mixed = tf.reduce_sum(tf.multiply(px1x2, theta), 1)
        print('You are using %.0f mixtures' % mixtures)
        # at the beginning, some errors are exactly zero.
        loss_seq = -tf.log(0.1*tf.maximum(px1x2x3_mixed, 1e-20))
        # loss_seq = -tf.log(0.1*px1x2x3_mixed, 1e-20))
        print(loss_seq)
        # loss_seq = decay * loss_seq
        # a_loss=tf.losses.mean_squared_error(labels=a_a, predictions=a)
        # v_loss=tf.losses.mean_squared_error(labels=a_v, predictions=v)
        # loss_seq0 =  loss_seq+a_loss+v_loss
        # print(loss_seq)
        # print(pad)
        print('gggg')

        print(loss_seq)
        # print(loss_seq)
        # loss_seq=loss_seq[:,:num]
        # print(loss_seq.shape)
        cost_seq = tf.reduce_mean(loss_seq)


    return cost_seq,mu1,mu2,s1,s2,rho,theta

class G_TCN_MDN_roll_abanden(object):
#用来产生概率分布
    def __init__(self):
        self.name='G_TCN_MDN_roll_abanden'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout,labels,pre_length):
        with tf.variable_scope(self.name) as vs:
            cost, mu1, mu2, s1, s2, rho, theta = TCN_MDN_roll_abanden(z, 100, channel_sizes, seq_len, kernel_size=kernel_size,
                                                         dropout=dropout, mixtures=1, labels=labels,pre_length=pre_length)
            return cost, mu1, mu2, s1, s2, rho, theta
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


def TCN_MDN_roall(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout,mixtures,labels,pre_length,pad):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length,
                          kernel_size=kernel_size, dropout=dropout)
    print(tcn)
    print('fffffff')
    batch_size=int(input_layer.shape[0])
    nhid=int(tcn.shape[2])
    tcn=tf.transpose(tcn,[0,2,1])
    tcn=tf.reshape(tcn,[tcn.shape[0]*tcn.shape[1],tcn.shape[2]])
    # tcn=tf.transpose(tcn,[1,0])

    linear = tf.contrib.layers.fully_connected(tcn, int(pre_length), activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01))
    # linear=tf.transpose(linear,[1,0])
    linear=tf.reshape(linear,[batch_size*nhid*pre_length])
    linear=tf.reshape(linear,[batch_size,nhid,pre_length])
    linear=tf.transpose(linear,[0,2,1])
    print(linear)
    # linear=tf.reshape(linear,[tcn.shape[0],pre_length,output_size])
    # linear=tf.transpose(linear,[0,2,1])
    with tf.name_scope("Output_MDN") as scope:
        params =6  # 7+theta
        # Two for distribution over hit&miss, params for distribution parameters
        output_units = mixtures * params
        W_o = tf.Variable(tf.random_normal(
            [output_size, output_units], stddev=0.01))
        print('ddddddd')
        b_o = tf.Variable(tf.constant(0.1, shape=[output_units]))
        outputs_tensor = linear
        print('outputs_tensor')
        print(outputs_tensor.shape)
        # is of size [batch_size*seq_len by output_units]
        # h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)
        h_out_tensor=tf.contrib.layers.fully_connected(outputs_tensor, output_units, activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01),biases_initializer=tf.zeros_initializer())

    with tf.name_scope('MDN_over_next_vector') as scope:
        # Next two lines are rather ugly, But its the most efficient way to
        # reshape the data
        print('hhhh')
        print(h_out_tensor)
        # h_xyz = tf.reshape(h_out_tensor, (pre_length, batch_size, output_units))
        h_xyz = tf.reshape(h_out_tensor, [batch_size,pre_length,output_units])
        # transpose to [batch_size, output_units, sl-1]
        # h_xyz = tf.transpose(h_xyz, [1, 2, 0])
        h_xyz = tf.transpose(h_xyz, [0, 2, 1])
        xn1=labels[:,:,0]
        xn2=labels[:,:,1]
        print('liyangggg')
        print(xn1)
        print(labels)
        # xn1=tf.reshape(xn1,[xn1.shape[0],xn1.shape[1]])
        # xn1=tf.transpose(xn1,[0,2,1])
        print('shape')
        print(xn1)
        # xn2 = tf.reshape(xn2, [xn2.shape[0], xn2.shape[1]])
        # xn2 = tf.transpose(xn2, [0, 2, 1])
        print('h_xyz')
        print(h_xyz)
        mu1, mu2, s1, s2,  rho, theta = tf.split(value=h_xyz, num_or_size_splits=params, axis=1)
        print(mu1)
        mu1=tf.reshape(mu1,[mu1.shape[0],pre_length])
        mu2=tf.reshape(mu2,[mu2.shape[0],pre_length])
        print('mu1')
        print(mu1)
        print(xn1)
        print(s1)

        # m   ke the theta mixtures
        # softmax all the theta's:
        max_theta = tf.reduce_max(theta, 1, keep_dims=True)
        theta = tf.subtract(theta, max_theta)
        theta = tf.exp(theta)
        normalize_theta = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
        theta = tf.multiply(normalize_theta, theta)

        # Deviances are non-negative and tho between -1 and 1
        s1 = tf.exp(s1)
        s2 = tf.exp(s2)
        # self.s3 = tf.exp(self.s3)
        rho = tf.tanh(rho)

        # probability in x1x2 plane
        px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2,s1, s2, rho)
        # px3 = tf_1d_normal(xn3, self.mu3, self.s3)
        # px1x2x3 = tf.multiply(px1x2, px3)

        # Sum along the mixtures in dimension 1
        px1x2x3_mixed = tf.reduce_sum(tf.multiply(px1x2, theta), 1)
        print(px1x2x3_mixed)
        print('You are using %.0f mixtures' % mixtures)
        # at the beginning, some errors are exactly zero.
        loss_seq = -tf.log(0.1*tf.maximum(px1x2x3_mixed, 1e-20))
        pad0 = tf.reshape(pad, [pad.shape[0], pad.shape[1]])

        #gram_loss
        mu10=tf.reshape(mu1,[batch_size,pre_length,1])
        mu1t=tf.reshape(mu1,[batch_size,1,pre_length])
        mu1_gram=tf.matmul(mu10,mu1t)
        print(mu1_gram)
        xn10 = tf.reshape(xn1,[batch_size,pre_length,1])
        xn1t = tf.reshape(xn1,[batch_size,1,pre_length])
        xn1_gram=tf.matmul(xn10,xn1t)
        x_gram=mu1_gram-xn1_gram
        x_gram_loss=tf.linalg.norm(x_gram,ord=2)
        x_gram_loss=pad0*x_gram_loss
        x_gram_loss=tf.reduce_sum(x_gram_loss)
        print(x_gram_loss)
        print('gaoyiba')
        #y的gram_loss

        mu20 = tf.reshape(mu2, [batch_size, pre_length, 1])
        mu2t = tf.reshape(mu2, [batch_size, 1, pre_length])
        mu2_gram = tf.matmul(mu20, mu2t)
        print(mu2_gram)
        xn20 = tf.reshape(xn2, [batch_size, pre_length, 1])
        xn2t = tf.reshape(xn2, [batch_size, 1, pre_length])
        xn2_gram = tf.matmul(xn20, xn2t)
        y_gram = mu2_gram - xn2_gram
        y_gram_loss = tf.linalg.norm(y_gram, ord=2)
        y_gram_loss=pad0*y_gram_loss
        y_gram_loss=tf.reduce_sum(y_gram_loss)
        x_gram_loss=x_gram_loss+y_gram_loss
        print(loss_seq)

        #加入有效区域的限制

        loss_seq = pad0 * loss_seq
        print(loss_seq)
        num = tf.count_nonzero(pad0, axis=1)
        sum = tf.reduce_sum(num)

        sum = tf.cast(sum, tf.float32)
        cost_seq = tf.reduce_sum(loss_seq) / sum
        x_gram_loss=x_gram_loss/sum
        # cost_seq=loss_seq
        # cost_seq = tf.reduce_mean(loss_seq)+0.02*x_gram_loss
        cost_seq = cost_seq+0.002*x_gram_loss
    return cost_seq,mu1,mu2,s1,s2,rho,theta,x_gram_loss

class G_TCN_MDN_roll(object):
#用来产生概率分布
    def __init__(self):
        self.name='G_TCN_MDN_roll'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout,labels,pre_length,pad):
        with tf.variable_scope(self.name) as vs:
            cost, mu1, mu2, s1, s2, rho, theta,x_gram_loss = TCN_MDN_roall(z, 100, channel_sizes, seq_len, kernel_size=kernel_size,
                                                         dropout=dropout, mixtures=1, labels=labels,pre_length=pre_length,pad=pad)
            return cost, mu1, mu2, s1, s2, rho, theta,x_gram_loss
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

def LSTM_MDN_roall(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout,mixtures,labels,pre_length):

    num_layers=3
    batch_size=int(input_layer.shape[0])
    input_layer=tf.transpose(input_layer,[0,2,1])
    with tf.name_scope("LSTM") as scope:
      cell = tf.nn.rnn_cell.MultiRNNCell([
        lstm_cell(output_size) for _ in range(num_layers)
      ])


      inputs = tf.unstack(input_layer, axis=2)
      # outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)

      print('len(inputs)')
      print(inputs)
      print(len(outputs))
      print(outputs)

    with tf.name_scope("Output_MDN") as scope:
        params =6  # 7+theta
        # Two for distribution over hit&miss, params for distribution parameters
        output_units = mixtures * params
        output = outputs[:-1]
        print('oooo')
        len_out=len(output)
        print(len(output))
        print(output)
        outputs_tensor = tf.concat(output, axis=0)
        print(outputs_tensor)
        print('ot')
        outputs_tensor=tf.reshape(outputs_tensor,[len_out,batch_size,output_size])
        outputs_tensor=tf.transpose(outputs_tensor,[1,2,0])
        print(outputs_tensor)
        # is of size [batch_size*seq_len by output_units]
        # h_out_tensor = tf.nn.xw_plus_b(outputs_tensor, W_o, b_o)
        h_out_tensor=tf.contrib.layers.fully_connected(outputs_tensor, pre_length, activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01),biases_initializer=tf.zeros_initializer())

    with tf.name_scope('MDN_over_next_vector') as scope:
        # Next two lines are rather ugly, But its the most efficient way to
        # reshape the data
        print('hhhh')
        print(h_out_tensor)
        # h_xyz = tf.reshape(h_out_tensor, (pre_length, batch_size, output_units))
        h_xyz = tf.transpose(h_out_tensor,[0,2,1])
        h_xyz=tf.contrib.layers.fully_connected(h_xyz, output_units, activation_fn=tf.nn.tanh,weights_initializer=tf.random_normal_initializer(0, 0.01),biases_initializer=tf.zeros_initializer())

        print(h_xyz)# transpose to [batch_size, output_units, sl-1]
        # h_xyz = tf.transpose(h_xyz, [1, 2, 0])
        h_xyz = tf.transpose(h_xyz, [0, 2, 1])
        xn1=labels[:,:,0]
        xn2=labels[:,:,1]
        print('liyangggg')
        print(xn1)
        print(labels)
        # xn1=tf.reshape(xn1,[xn1.shape[0],xn1.shape[1]])
        # xn1=tf.transpose(xn1,[0,2,1])
        print('shape')
        print(xn1)
        # xn2 = tf.reshape(xn2, [xn2.shape[0], xn2.shape[1]])
        # xn2 = tf.transpose(xn2, [0, 2, 1])
        print('h_xyz')
        print(h_xyz)
        mu1, mu2, s1, s2,  rho, theta = tf.split(value=h_xyz, num_or_size_splits=params, axis=1)
        print(mu1)
        mu1=tf.reshape(mu1,[mu1.shape[0],pre_length])
        mu2=tf.reshape(mu2,[mu2.shape[0],pre_length])
        print('mu1')
        print(mu1)
        print(xn1)

        # m   ke the theta mixtures
        # softmax all the theta's:
        max_theta = tf.reduce_max(theta, 1, keep_dims=True)
        theta = tf.subtract(theta, max_theta)
        theta = tf.exp(theta)
        normalize_theta = tf.reciprocal(tf.reduce_sum(theta, 1, keep_dims=True))
        theta = tf.multiply(normalize_theta, theta)

        # Deviances are non-negative and tho between -1 and 1
        s1 = tf.exp(s1)
        s2 = tf.exp(s2)
        # self.s3 = tf.exp(self.s3)
        rho = tf.tanh(rho)

        # probability in x1x2 plane
        px1x2 = tf_2d_normal(xn1, xn2, mu1, mu2,s1, s2, rho)
        # px3 = tf_1d_normal(xn3, self.mu3, self.s3)
        # px1x2x3 = tf.multiply(px1x2, px3)

        # Sum along the mixtures in dimension 1
        px1x2x3_mixed = tf.reduce_sum(tf.multiply(px1x2, theta), 1)
        print(px1x2x3_mixed)
        print('You are using %.0f mixtures' % mixtures)
        # at the beginning, some errors are exactly zero.
        loss_seq = -tf.log(0.1*tf.maximum(px1x2x3_mixed, 1e-20))

        #gram_loss
        mu10=tf.reshape(mu1,[batch_size,pre_length,1])
        mu1t=tf.reshape(mu1,[batch_size,1,pre_length])
        mu1_gram=tf.matmul(mu10,mu1t)
        print(mu1_gram)
        xn10 = tf.reshape(xn1,[batch_size,pre_length,1])
        xn1t = tf.reshape(xn1,[batch_size,1,pre_length])
        xn1_gram=tf.matmul(xn10,xn1t)
        x_gram=mu1_gram-xn1_gram
        x_gram_loss=tf.linalg.norm(x_gram,ord=2)
        x_gram_loss=tf.reduce_mean(x_gram_loss)
        print(x_gram_loss)
        print('gaoyiba')
        #y的gram_loss
        mu20 = tf.reshape(mu2, [batch_size, pre_length, 1])
        mu2t = tf.reshape(mu2, [batch_size, 1, pre_length])
        mu2_gram = tf.matmul(mu20, mu2t)
        print(mu2_gram)
        xn20 = tf.reshape(xn2, [batch_size, pre_length, 1])
        xn2t = tf.reshape(xn2, [batch_size, 1, pre_length])
        xn2_gram = tf.matmul(xn20, xn2t)
        y_gram = mu2_gram - xn2_gram
        y_gram_loss = tf.linalg.norm(y_gram, ord=2)
        y_gram_loss=tf.reduce_mean(y_gram_loss)
        x_gram_loss=x_gram_loss+y_gram_loss
        print(loss_seq)
        # cost_seq=loss_seq
        # cost_seq = tf.reduce_mean(loss_seq)+0.02*x_gram_loss
        cost_seq = tf.reduce_mean(loss_seq)
    return cost_seq,mu1,mu2,s1,s2,rho,theta,x_gram_loss

class G_LSTM_MDN_roll(object):
#用来产生概率分布
    def __init__(self):
        self.name='G_LSTM_MDN_roll'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout,labels,pre_length):
        with tf.variable_scope(self.name) as vs:
            cost, mu1, mu2, s1, s2, rho, theta,x_gram_loss = LSTM_MDN_roall(z, 64, channel_sizes, seq_len, kernel_size=kernel_size,
                                                         dropout=dropout, mixtures=1, labels=labels,pre_length=pre_length)
            return cost, mu1, mu2, s1, s2, rho, theta,x_gram_loss
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_TCN(object):
    def __init__(self):
        self.name='G_TCN'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout):
        with tf.variable_scope(self.name) as vs:
            outputs = TCN_G(z, 2, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout)
            return outputs
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class G_TCN_roll(object):
    def __init__(self):
        self.name='G_TCN_roll'
    def __call__(self, z,channel_sizes,seq_len,kernel_size,dropout,pre_length):
        with tf.variable_scope(self.name) as vs:
            outputs = TCN_G_roll(z, 2, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout,pre_length=pre_length)
            return outputs
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_TCN(object):
    def __init__(self):
        self.name = "D_TCN"

    def __call__(self, x,channel_sizes,seq_len,kernel_size,dropout, reuse=False):
        print('xx')
        print(x)
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            out_label = TCN_D(x, 1, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout)

        return out_label
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_TCN(object):
    def __init__(self):
        self.name = 'C_TCN'

    def __call__(self, x,channel_sizes,seq_len,kernel_size,dropout, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out_label = TCN_C(x, 3, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout)
            return out_label

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class C_TCN_new(object):
    def __init__(self):
        self.name = 'C_TCN_new'

    def __call__(self, x,channel_sizes,seq_len,kernel_size,dropout, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            out_label = TCN_C(x, 3, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout)
            out_label=tf.reshape(out_label,[out_label.shape[0],out_label.shape[1]*out_label.shape[2]])
            out_label = tcl.fully_connected(out_label, 3, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            print('xiaozhenga')
            print(out_label)
            return out_label

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

# class C_TCN_new(object):
#     def __init__(self):
#         self.name = 'C_TCN_new'
#
#     def __call__(self, x,channel_sizes,seq_len,kernel_size,dropout, reuse=False):
#         with tf.variable_scope(self.name) as scope:
#             if reuse:
#                 scope.reuse_variables()
#             out_label = TCN_C(x, 100, channel_sizes, seq_len, kernel_size=kernel_size, dropout=dropout)
#             out_label=tf.reshape(out_label,[x.shape[0],100*seq_len],name=None)
#             print('haoran',out_label)
#             d = tcl.fully_connected(out_label, 100, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
#             out = tcl.fully_connected(d, 3, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
#             return out,d
#
#     @property
#     def vars(self):
#         return [var for var in tf.global_variables() if self.name in var.name]

class G_mlp(object):
    def __init__(self):
        self.name = 'G_mlp'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, 4 * 4 * 512, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            g = tcl.fully_connected(g, 64*64*3, activation_fn=tf.nn.tanh, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, tf.stack([tf.shape(z)[0], 64, 64, 3]))
            return g
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class D_mlp(object):
    def __init__(self):
        self.name = "D_mlp"

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()
            d = tcl.fully_connected(tf.flatten(x), 64, activation_fn=tf.nn.relu,normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            d = tcl.fully_connected(d, 64,activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            logit = tcl.fully_connected(d, 1, activation_fn=None)

        return logit

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

#-------------------------------- MNIST for test ------
class G_mlp_mnist(object):
    def __init__(self):
        self.name = "G_mlp_mnist"
        self.X_dim = 784

    def __call__(self, z):
        with tf.variable_scope(self.name) as vs:
            g = tcl.fully_connected(z, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(g, self.X_dim, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer(0, 0.02))
        return g

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_mlp_mnist():
    def __init__(self):
        self.name = "D_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))

            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes

        return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class Q_mlp_mnist():
    def __init__(self):
        self.name = "Q_mlp_mnist"

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            shared = tcl.fully_connected(x, 128, activation_fn=tf.nn.relu, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 10, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02)) # 10 classes
        return q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


###############################################  conv #############################################
class G_conv(object):
    def __init__(self):
        self.name = 'G_conv'
        self.size = 4
        self.channel = 3

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            g = tcl.fully_connected(z, self.size * self.size * 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            g = tf.reshape(g, (-1, self.size, self.size, 1024))  # size
            g = tcl.conv2d_transpose(g, 512, 3, stride=2, # size*2
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 256, 3, stride=2, # size*4
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 128, 3, stride=2, # size*8
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))

            g = tcl.conv2d_transpose(g, self.channel, 3, stride=2, # size*16
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv(object):
    def __init__(self):
        self.name = 'D_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=4, # 4x4x512
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.flatten(shared)

            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 2, activation_fn=None) # 10 classes
            return d, q

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv(object):
    def __init__(self):
        self.name = 'C_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            #d = tcl.conv2d(d, num_outputs=size * 8, kernel_size=3, # 4x4x512
            #			stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)

            q = tcl.fully_connected(tcl.flatten(shared), 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 10, activation_fn=None) # 10 classes

            return q
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

class V_conv(object):
    def __init__(self):
        self.name = 'V_conv'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx64x64x3 -> bzx32x32x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 16x16x128
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 4, kernel_size=4, # 8x8x256
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            shared = tcl.conv2d(shared, num_outputs=size * 8, kernel_size=3, # 4x4x512
                        stride=2, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)

            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)

            v = tcl.fully_connected(tcl.flatten(shared), 128)
            return v
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]


# -------------------------------- MNIST for test
class G_conv_mnist(object):
    def __init__(self):
        self.name = 'G_conv_mnist'

    def __call__(self, z):
        with tf.variable_scope(self.name) as scope:
            #g = tcl.fully_connected(z, 1024, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
            #						weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.fully_connected(z, 7*7*128, activation_fn = tf.nn.relu, normalizer_fn=tcl.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tf.reshape(g, (-1, 7, 7, 128))  # 7x7
            g = tcl.conv2d_transpose(g, 64, 4, stride=2, # 14x14x64
                                    activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            g = tcl.conv2d_transpose(g, 1, 4, stride=2, # 28x28x1
                                        activation_fn=tf.nn.sigmoid, padding='SAME', weights_initializer=tf.random_normal_initializer(0, 0.02))
            return g
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class D_conv_mnist(object):
    def __init__(self):
        self.name = 'D_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=4, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=lrelu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=4, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.flatten(shared)

            d = tcl.fully_connected(shared, 1, activation_fn=None, weights_initializer=tf.random_normal_initializer(0, 0.02))
            q = tcl.fully_connected(shared, 128, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            q = tcl.fully_connected(q, 10, activation_fn=None) # 10 classes
            return d, q
    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

class C_conv_mnist(object):
    def __init__(self):
        self.name = 'C_conv_mnist'

    def __call__(self, x, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            size = 64
            shared = tcl.conv2d(x, num_outputs=size, kernel_size=5, # bzx28x28x1 -> bzx14x14x64
                        stride=2, activation_fn=tf.nn.relu)
            shared = tcl.conv2d(shared, num_outputs=size * 2, kernel_size=5, # 7x7x128
                        stride=2, activation_fn=lrelu, normalizer_fn=tcl.batch_norm)
            shared = tcl.fully_connected(tcl.flatten( # reshape, 1
                        shared), 1024, activation_fn=tf.nn.relu)

            #c = tcl.fully_connected(shared, 128, activation_fn=tf.nn.relu, normalizer_fn=tcl.batch_norm)
            c = tcl.fully_connected(shared, 10, activation_fn=None) # 10 classes
            return c
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

