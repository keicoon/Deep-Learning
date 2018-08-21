import tensorflow as tf

class nalu():
    def __init__(self, num_inputs, num_outputs, use_gating=True):
        self.EPSILON = 1e-7
        self.use_gating = use_gating
        shape = (num_inputs, num_outputs)

        self.W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
        self.M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
        self.G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    
    def __call__(self, input):
        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        m = tf.exp(tf.matmul(tf.log(tf.abs(input) + EPSILON), W))
        a = tf.matmul(input, W)
        
        if self.use_gating:
            g = tf.sigmoid(tf.matmul(input, self.G))
            out = g * a + (1 - g) * m
        else
            out = a + m

        return out