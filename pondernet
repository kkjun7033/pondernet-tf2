import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def recurrent_m(n_seq, n_hidden):
    x_input = tf.keras.Input(shape=(n_seq, n_hidden)) 
    x_out = tf.keras.layers.LSTM(n_hidden)(x_input)
    x_out = tf.keras.layers.BatchNormalization()(x_out)   

    model = tf.keras.Model(inputs = x_input, outputs = x_out)
    return model

def lambda_m(n_hidden):
    x_input = tf.keras.Input(shape=(n_hidden)) 
    x_out = tf.keras.layers.Dense(1, 'sigmoid')(x_input)    
    
    model = tf.keras.Model(inputs = x_input, outputs = x_out)
    return model
    
def out_m(n_hidden):
    x_input = tf.keras.Input(shape=(n_hidden)) 
    x_out = tf.keras.layers.Dense(190)(x_input) 
    x_out = tf.keras.layers.BatchNormalization()(x_out)
    x_out = tf.keras.layers.Activation('tanh')(x_out)
    
    model = tf.keras.Model(inputs = x_input, outputs = x_out)
    return model

def Ponder(data, train_step):
    p = []
    y = []
    lam =[]
    un_halted_prob = tf.ones((tf.shape(data)[0],1))
    halted = tf.ones((tf.shape(data)[0],1))
    
    h = LSTM_model(data)
    for n in range(train_step):
        lambda_n = lambda_model(h)#[1])
        y_n = out_model(h)#[1])

        p_n = un_halted_prob * lambda_n #(1-p)p
        un_halted_prob = un_halted_prob * (1 - lambda_n) #update
        
        lam.append(lambda_n)
        p.append(p_n)
        y.append(y_n)
        
        h = LSTM_model(tf.expand_dims(h,1))
        
        ps = tf.stack(p, 1) 
        psd = tf.expand_dims(tf.reduce_sum(ps,1),-1)
        psd = ps/psd
    return psd, tf.stack(y, 1), tf.stack(lam, 1)

def ReconstructionLoss(pro, y_h, y):
    total_loss = 0
    y = tf.cast(y, tf.float32)
    for n in range(tf.shape(pro)[1]):
        loss = tf.transpose(pro[:,n,:]) * tf.keras.losses.mean_squared_error(y_h[:,n,:], y)
        total_loss += loss
    return tf.reduce_mean(total_loss)

def RegularizationLoss(p, lamb):
    priors = tf.repeat(prior_m(lamb, max_train_step), [tf.shape(p)[0]], axis = 0)
    loss = tf.keras.losses.KLDivergence()(priors, p)
    return loss


def prior_m(lamb, max_train_step):
    pp=lamb
    li=[]
    for k in range(max_train_step):
        li.append((1-pp)**float(k-1) * pp)
    pri = li/np.sum(li)
    prior = np.reshape(pri, (1,max_train_step,1))
    return prior
    
 
