#### A tensorflow implementation of NAL(neural accumulator) and NALU(Neural Arithmetic Logic Units)

### Code snippet
```
from nac import nac
# from nalu import nalu

X = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2])
Y_true = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 1])

nacLayer = nac(2, 1)

Y_pred = nacLayer(X)

loss = tf.nn.l2_loss(Y_pred - Y_true)
```

### Reference
- info from [paper](https://arxiv.org/pdf/1808.00508.pdf)
- source from [github](https://github.com/grananqvist/NALU-tf/)