# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# File       : splite_tcn_gan.py
# Time       ：2022/5/25 1:15
# Author     ：J ▄︻┻┳═一
# version    ：python 3.6
# Description：
"""
import tensorflow as tf
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, sys
import pickle

sys.path.append('utils')
from nets import *
from datas import *
batch_size = 100
def index_generator(n_train, batch_size):
    all_indices = np.arange(n_train)
    start_pos = 0
    #while True:
    #    all_indices = np.random.permutation(all_indices)
    for batch_idx, batch in enumerate(range(start_pos, n_train, batch_size)):

        start_ind = batch
        end_ind = start_ind + batch_size

        # last batch
        if end_ind > n_train:
            diff = end_ind - n_train
            toreturn = all_indices[start_ind:end_ind]
            toreturn = np.append(toreturn, all_indices[0:diff])
            start_pos = diff
            yield batch_idx + 1, toreturn
            break

        yield batch_idx + 1, all_indices[start_ind:end_ind]

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


# for test
def sample_y(m, n, ind):
    y = np.zeros([m, n])
    for i in range(m):
        y[i, i % 8] = 1
    # y[:,7] = 1
    # y[-1,0] = 1
    return y


def label2vec(label_list,len_l):
    res = []
    for l in label_list:
        ll = [0] * 3
        ll[l] = 1
        lll = [ll]
        res.append(lll)
    res0 = np.array(res)
    return res0


def concat(z, y):
    return tf.concat([z, y], 2)


class CGAN_Classifier(object):
    def __init__(self, generator, discriminator, classifier,g_tcn_mdn, X_train, Y_train, c_vec):
        self.generator = generator
        self.discriminator = discriminator
        self.classifier = classifier
        self.g_tcn_mdn=g_tcn_mdn

        # data
        self.X_train = X_train
        self.y_train = Y_train


        seq_len = 100
        in_channels = 4
        # nhid = 10
        nhid = 64
        # levels = 8
        levels = 5
        channel_sizes = [nhid] * levels
        # kernel_size = 8
        kernel_size = 4
        dropout = 0

        # self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        self.z = tf.placeholder(tf.float32, (batch_size, seq_len, in_channels))
        # self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.X = tf.placeholder(tf.float32, (batch_size, pre_len, 2))
        self.y = tf.placeholder(tf.float32, (batch_size, 1,3))
        self.p= tf.placeholder(tf.float32, (batch_size, pre_len, 1))

        # self.X = tf.placeholder(tf.float32, shape=[None, self.size, self.size, self.channel])
        # self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        # self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # nets
        # self.G_sample = self.generator(concat(self.z, self.y), channel_sizes, seq_len, kernel_size, dropout)
        # print('gggg')
        # print(self.G_sample)
        self.mdn_cost, self.mu1, self.mu2, self.s1, self.s2, self.rho, self.theta,self.x_gram_loss = self.g_tcn_mdn(self.z, channel_sizes, seq_len, kernel_size, dropout,self.X,pre_len,self.p)
        print('mu1')
        print(self.mu1)
        self.mu_xy=tf.stack([self.mu1,self.mu2],axis=2)
        # self.mu_xy=tf.transpose(self.mu_xy,[0,2,1])
        print(self.mu_xy)

        self.D_real = self.discriminator(self.X, channel_sizes, pre_len, kernel_size, dropout)
        self.D_fake = self.discriminator(self.mu_xy, channel_sizes, pre_len, kernel_size, dropout, reuse=True)

        self.C_real = self.classifier(self.X, channel_sizes, pre_len, kernel_size, dropout)
        self.C_fake = self.classifier(self.mu_xy, channel_sizes, pre_len, kernel_size, dropout, reuse=True)

        # loss
        # self.D_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(
        #     self.D_real))) + tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.D_loss = tf.reduce_mean(-tf.reduce_sum(tf.ones_like(self.D_real) * tf.log(self.D_real), reduction_indices=[1]))\
                      +tf.reduce_mean(-tf.reduce_sum(tf.zeros_like(self.D_fake) * tf.log(self.D_fake), reduction_indices=[1]))
        # self.G_loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))
        self.G_loss = tf.reduce_mean(-tf.reduce_sum(tf.ones_like(self.D_fake) * tf.log(self.D_fake), reduction_indices=[1]))

        self.C_real_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.C_real, labels=self.y))  # real label
        self.C_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.C_fake, labels=self.y))
        m=tf.losses.mean_squared_error(labels=self.X, predictions=self.mu_xy)
        print('xiaozheng',self.X)
        self.actual_X = self.X * self.p
        self.actual_mu_xy = self.mu_xy * self.p

        # self.T_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.X[:,:,:2], predictions=self.mu_xy[:,:,:2]))
        self.T_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.actual_X, predictions=self.actual_mu_xy))

        #learning rate
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(5e-4, global_step, 1400, 0.95, staircase=True)

        # solver
        self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                                                                       var_list=self.discriminator.vars)
        # self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss + self.C_fake_loss, var_list=self.generator.vars)
        self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss+self.C_fake_loss+0.1*self.mdn_cost+0.0002*self.x_gram_loss,
                                                                                       var_list=self.g_tcn_mdn.vars)
        self.C_real_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_real_loss,
                                                                                            var_list=self.classifier.vars)
        # self.C_fake_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.C_fake_loss, var_list=self.generator.vars)

        self.saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    def train(self, sample_dir,wf, ckpt_dir='save_model_roll_r10p20_more', training_epoches=10000):
        wf = open(wf, 'w')
        fig_count = 0
        n_train = X_train.shape[0]
        #n_train = 100
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        for epoch in range(training_epoches):
            ii=0

            for batch_idx, indices in index_generator(n_train, batch_size):
                # print(batch_idx)
                x_data = X_train[indices]
                ii+=1
                # print('xxxxxxxx')
                # print(x.shape)
                # x=np.transpose(x, [0,2,1])
                y_data = Y_train[indices]
                c_data=intent_label[indices]
                pad_data=padding[indices]
                # if ii==2:
                #     break
            #
            # for epoch in range(training_epoches):
            #     # update D
            #     ccc =x_data[x_data[:,:,2]>1]
            #     print('ccc')
            #     print(ccc)
                for _ in range(1):
                    # X_b, y_b = self.data(batch_size)
                    _=self.sess.run(
                        self.D_solver,
                        feed_dict={self.X: y_data, self.y: c_data, self.z: x_data}
                    )
                    # print(d)
                    # print(r)
                    # print(dl)
                # update G
                for _ in range(2):
                    self.sess.run(
                        self.G_solver,
                        feed_dict={self.y: c_data, self.z: x_data, self.X: y_data,self.p:pad_data}
                    )
                # update C
                for _ in range(1):
                    # real label to train C
                    self.sess.run(
                        self.C_real_solver,
                        feed_dict={self.X: y_data, self.y: c_data})
                '''
                    # fake img label to train G
                    self.sess.run(
                        self.C_fake_solver,
                        feed_dict={self.y: y_b, self.z: sample_z(batch_size, self.z_dim)})
                '''
            # save img, model. print loss
            if epoch % 1 == 0 or epoch < 100:
                D_loss_curr, C_real_loss_curr = self.sess.run(
                    [self.D_loss, self.C_real_loss],
                    feed_dict={self.X: y_data, self.y: c_data, self.z: x_data})
                G_loss_curr, C_fake_loss_curr,Logloss, T_loss_curr,x_gram_loss = self.sess.run(
                    [self.G_loss, self.C_fake_loss,self.mdn_cost, self.T_loss,self.x_gram_loss],
                    feed_dict={self.y: c_data, self.z: x_data, self.X: y_data,self.p:pad_data})

                print(
                    'Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4};Logloss: {:.4};T_loss: {:.5};T_loss: {:.5}'.format(
                        epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr,Logloss, T_loss_curr,x_gram_loss))
                txt_word = 'Iter: {}; D loss: {:.4}; G_loss: {:.4}; C_real_loss: {:.4}; C_fake_loss: {:.4};Logloss: {:.4};T_loss: {:.5}'.format(
                        epoch, D_loss_curr, G_loss_curr, C_real_loss_curr, C_fake_loss_curr,Logloss, T_loss_curr)
                wf.write(txt_word + '\n')
                if epoch % 10 == 0:
                    mu_xy,s1,s2,rho,theta=self.sess.run([self.mu_xy,self.s1, self.s2, self.rho, self.theta],feed_dict={self.z: x_data})
                    #print(s1)
                    dic={}
                    dic['true']=y_data
                    dic['pre']=mu_xy
                    dic['pad']=pad_data
                    data_output = open('mu_xy_tcn0714p20_more.pkl', 'wb')
                    pickle.dump(dic, data_output)
                    data_output.close()

                    self.saver.save(self.sess, os.path.join(ckpt_dir, "save_model_roll_r10p20_more.ckpt"))
                if T_loss_curr < 0.0001 or epoch==training_epoches:
                    print(epoch)
                    mu_xy = self.sess.run(self.mu_xy, feed_dict={self.z: x_data})
                    dic = {}
                    dic['true'] = y_data
                    dic['pre'] = mu_xy
                    dic['pad'] = pad_data
                    data_output = open('mu_xy_last_p20_0714_more.pkl', 'wb')
                    pickle.dump(dic, data_output)
                    data_output.close()
                    self.saver.save(self.sess, os.path.join(ckpt_dir, "save_model_roll_r10p20_more.ckpt"))
                    # pre = self.sess.run(self.G_sample, feed_dict={self.y: c_vec, self.z: X_train, self.X: Y_train})
                    l, m1, m2, ss1, ss2, r0, t0 = self.sess.run([self.mdn_cost, self.mu1, self.mu2, self.s1, self.s2, self.rho, self.theta], feed_dict={self.y: c_vec, self.z: X_train, self.X: Y_train,self.p:pad_data})
                    if True:
                        d = {}
                        print(l)
                        print(m1)
                        print(m1.shape)
                        print(np.max(m1))
                        print(np.min(m1))

                        print(np.max(m2))
                        print(np.min(m2))
                        d['m1'] = m1
                        d['m2'] = m2
                        d['ss1'] = ss1
                        d['ss2'] = ss2
                        d['r0'] = r0
                        d['t0'] = t0
                        data_output = open('cgan_d_mdn_0521.pkl', 'wb')
                        pickle.dump(d, data_output)
                        data_output.close()
                        print('11')
                        print(Y_train)
                        break
                        # print(m2)
                        # print('11')
                        # print(ss1)
                        # print('11')
                        # print(ss2)
                        # break

                    # break


                    # if epoch % 1000 == 0:
                    # 	y_s = sample_y(16, self.y_dim, fig_count%10)
                    # 	samples = self.sess.run(self.G_sample, feed_dict={self.y: y_s, self.z: sample_z(16, self.z_dim)})
                    #
                    # 	fig = self.data.data2fig(samples)
                    # 	plt.savefig('{}/{}_{}.png'.format(sample_dir, str(fig_count).zfill(3), str(fig_count%10)), bbox_inches='tight')
                    # 	fig_count += 1
                    # 	plt.close(fig)
                    #
                    # if epoch % 2000 == 0:
                    # 	self.saver.save(self.sess, os.path.join(ckpt_dir, "cgan_classifier.ckpt"))


if __name__ == '__main__':

    from dataloader import *
    #从pkl读取数据
    import pickle

    data_output = open('./roll_data/train_data/r1p20_start.pkl', 'rb')
    DATA = pickle.load(data_output)
    # print(DATA)
    X_train=DATA['x']
    Y_train=DATA['y'][:,:,:2]
    intent=DATA['intent']
    padding=DATA['padding']
    intent_label=label2vec(intent,Y_train.shape[1])

    # demo_x=X_train[0]
    # print(demo_x.shape)
    # demo_x[:,0]=demo_x[:,0]* (122.5535567 - 122.4512304) + 122.4512304
    # demo_x[:,1]=demo_x[:,1]* (31.01321725 - 30.9712419) + 30.9712419
    # demo_x[:,2]=demo_x[:,2]* (20) + 3
    # demo_x[:,3]=demo_x[:,3]* (360)
    # print('demo_x')
    # print(demo_x)
    #
    # demo_y = Y_train[0]
    # print(demo_x.shape)
    # demo_y[:, 0] = demo_y[:, 0] * (122.5535567 - 122.4512304) + 122.4512304
    # demo_y[:, 1] = demo_y[:, 1] * (31.01321725 - 30.9712419) + 30.9712419
    # # demo_y[:, 2] = demo_y[:, 2] * (20) + 3
    # # demo_y[:, 3] = demo_y[:, 3] * (360)
    # print('demo_y')
    # print(demo_y)

    mean=[122.5210562,30.98687026]
    std=[3.51145598e-02,2.10093706e-02]
    print(mean,std)
    demo_x = X_train[0,:,:2]
    demo_y = Y_train[0]
    demo_x=demo_x*std+mean
    demo_y=demo_y*std+mean
    print(demo_x)
    print('demo_y')
    print(demo_y)




    # DATA[:, :, 0] = DATA[:, :, 0] * (122.5535567 - 122.4512304) + 122.4512304
    # DATA[:, :, 1] = DATA[:, :, 1] * (31.01321725 - 30.9712419) + 30.9712419

    print('zheng')
    print(X_train.shape)
    print(Y_train.shape)
    pre_len=Y_train.shape[1]


    #
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    wf = r'.\log\log_cgan_mdn_0601_small.txt'


    # save generated images
    sample_dir = 'Samples/tcn_cgan_classifier0608_more'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # param
    generator = G_TCN()
    discriminator = D_TCN()
    classifier = C_TCN_new()
    g_tcn_mdn=G_TCN_MDN_roll()

    # data = mnist()

    # # run
    cgan_c = CGAN_Classifier(generator, discriminator, classifier,g_tcn_mdn, X_train, Y_train, intent_label)
    cgan_c.train(sample_dir,wf)
    #
