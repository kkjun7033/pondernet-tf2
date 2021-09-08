import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pondernet



LSTM_model = recurrent_m(1,190) #n_seq, n_hidden
lambda_model = lambda_m(190)
out_model = out_m(190)

train_variable = LSTM_model.trainable_variables + lambda_model.trainable_variables + out_model.trainable_variables
Adam = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean()

def Train(x, y, max_train_step, max_seq):
    un_halted_prob = tf.ones((tf.shape(x)[0],1))
    T_y = tf.zeros((1))
    reg_loss_in = tf.zeros((1))
    pi_out = []
    with tf.GradientTape() as tape:
        for i in range(1, max_sequ+1):
            LSTM_model.reset_states()
            p, y_h, _ = Ponder(tf.expand_dims(x[:,max_sequ-i,:],1), max_train_step)

            T_y += tf.reduce_sum(p * y_h, 1)
            reg_loss_in += RegularizationLoss(p, 0.65 - i*0.15)
        T_y = T_y/max_sequ 
        loss_t = tf.keras.losses.mean_squared_error(T_y, y)
        loss_t = tf.reduce_mean(loss_t)        

        loss = tf.cast(loss_t, tf.float32) + reg_loss_in
        print(loss)
        
    grads = tape.gradient(loss, train_variable)
    Adam.apply_gradients(zip(grads, train_variable))
    train_loss(loss)


max_sequ = 3
max_train_step = 5
to = [x for x in range(data)]

for n in range(1000):
    idx = random.sample(to, 1000) 
    Train(Train_X[idx], Train_Y[idx], max_train_step, max_sequ)
    train_loss.result()
    print(n)    
    
    
    
def Inference_ponder(x, step):
    h = LSTM_model(x)
    un_halted_prob = tf.ones((tf.shape(data)[0],1))     
    y_zip = []
    in_zip=[]    
    for i in range(step):   
        #LSTM_model.reset_states()
        lambda_n = lambda_model(h)
        y_n = out_model(h)                 
        halt = tfp.distributions.Bernoulli(probs=lambda_n).sample()
        h = LSTM_model(tf.expand_dims(h,1))        
     
        print('layer', i, lambda_n[0].numpy(), 'Stop', halt.numpy())
        if halt == 1:
            break
    return lambda_n, y_n, in_zip

def Inference(x, step, max_seq):
    y_zip = []
    out_zip = []
    for i in range(1, max_sequ+1):
        _, y_h, in_z = Inference_ponder(tf.expand_dims(x[:,max_sequ-i,:],1), max_train_step)
        y_zip.append(y_h) 
        print('seq', i+1, lam_out[0].numpy(), halt.numpy())
    return np.mean(np.array(y_zip), 0)   
