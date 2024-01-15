"""
@author: Maziar Raissi
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model
import time

class FBSNN(Model): # Forward-Backward Stochastic Neural Network
    def __init__(self, Xi, T,
                       M, N, D,
                       layers, phi_tf, g_tf, mu_tf, sigma_tf, learning_rate):
        super().__init__()
        self.Xi = Xi # initial point
        self.T = T # terminal time
        
        self.M = M # number of trajectories
        self.N = N # number of time snapshots   
        self.D = D # number of dimensions
        self.phi_tf = phi_tf # M x 1, M x D, M x 1, M x D
        self.g_tf = g_tf # M x D
        self.mu_tf = mu_tf # M x 1, M x D, M x 1, M x D
        self.sigma_tf = sigma_tf # M x 1, M x D, M x 1
        self.learning_rate = learning_rate
        
        # layers eg [100, 256, 256, 256, 256, 1] 
        self.layers = layers # (D+1) --> 1
        model = tf.keras.Sequential()
        for layer in layers[:-1]:
            model.add(Dense(layer, activation='sin', kernel_initializer='glorot_normal', bias_initializer='zeros'))
        model.add(Dense(layers[-1]))
        self.model = model
        
        # define keras dense layers
        self.t_tf = Input(shape=(M, N+1, 1), name='t_input') # M x (N+1) x 1
        self.W_tf = Input(shape=(M, N+1, D), name='W_input') # M x (N+1) x D
        self.Xi_tf = Input(shape=(1, D), name='Xi_input') # 1 x D

        self.loss, self.X_pred, self.Y_pred, self.Y0_pred = self.loss_function(self.t_tf, self.W_tf, self.Xi_tf)
        
        
    def call(self, inputs):
        return self.model(inputs)  
    
    def fetch_minibatch(self):
        T = self.T
        
        M = self.M
        N = self.N
        D = self.D
        
        Dt = np.zeros((M,N+1,1)) # M x (N+1) x 1
        DW = np.zeros((M,N+1,D)) # M x (N+1) x D
        
        dt = T/N
        
        Dt[:,1:,:] = dt
        DW[:,1:,:] = np.sqrt(dt)*np.random.normal(size=(M,N,D))
        
        t = np.cumsum(Dt,axis=1) # M x (N+1) x 1
        W = np.cumsum(DW,axis=1) # M x (N+1) x D
        
        return t, W
    
    def net_u(self, t, X): # M x 1, M x D
        
        u = self.model(tf.concat([t,X], 1), self.weights, self.biases) # M x 1
        Du = tf.gradients(u, X)[0] # M x D
        
        return u, Du

    def Dg_tf(self, X): # M x D
        return tf.gradients(self.g_tf(X), X)[0] # M x D
        
    def loss_function(self, t, W, Xi): # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = 0
        X_list = []
        Y_list = []
        
        t0 = t[:,0,:]
        W0 = W[:,0,:]
        X0 = tf.tile(Xi,[self.M,1]) # M x D
        Y0, Z0 = self.net_u(t0,X0) # M x 1, M x D
        
        X_list.append(X0)
        Y_list.append(Y0)
        
        for n in range(0,self.N):
            t1 = t[:,n+1,:]
            W1 = W[:,n+1,:]
            X1 = X0 + self.mu_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1)), axis=[-1])
            Y1_tilde = Y0 + self.phi_tf(t0,X0,Y0,Z0)*(t1-t0) + tf.reduce_sum(Z0*tf.squeeze(tf.matmul(self.sigma_tf(t0,X0,Y0),tf.expand_dims(W1-W0,-1))), axis=1, keepdims = True)
            Y1, Z1 = self.net_u(t1,X1)
            
            loss += tf.reduce_sum(tf.square(Y1 - Y1_tilde))
            
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1
            
            X_list.append(X0)
            Y_list.append(Y0)
            
        loss += tf.reduce_sum(tf.square(Y1 - self.g_tf(X1)))
        loss += tf.reduce_sum(tf.square(Z1 - self.Dg_tf(X1)))

        X = tf.stack(X_list,axis=1)
        Y = tf.stack(Y_list,axis=1)
        
        return loss, X, Y, Y[0,0,0]

    def train(self, N_Iter, learning_rate):
            
            start_time = time.time()
            for it in range(N_Iter):
                
                t_batch, W_batch = self.fetch_minibatch() # M x (N+1) x 1, M x (N+1) x D
                
                tf_dict = {self.Xi_tf: self.Xi, self.t_tf: t_batch, self.W_tf: W_batch, self.learning_rate: learning_rate}
                
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % 10 == 0:
                    elapsed = time.time() - start_time
                    loss_value, Y0_value, learning_rate_value = self.sess.run([self.loss, self.Y0_pred, self.learning_rate], tf_dict)
                    print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' % 
                        (it, loss_value, Y0_value, elapsed, learning_rate_value))
                    start_time = time.time()
                
