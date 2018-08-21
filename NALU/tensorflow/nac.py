import tensorflow as tf

class nac():
    def __init__(self, num_inputs, num_outputs):
        shape = (num_inputs, num_outputs)
        
        self.W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
        self.M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    
    def __call__(self, input):
        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        a = tf.matmul(input, W)
        return a