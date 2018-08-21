module.exports = (tf) => {
    const primaryCapsLayer = require('./primaryCapsLayer')(tf);
    const digitCapsLayer = require('./digitCapsLayer')(tf);

    return class CapsNet {
        constructor() {
            const model = tf.sequential();
            model.add(tf.layers.conv2d({
                inputShape: [28, 28, 1],
                filters: 256,
                kernelSize: 9,
                strides: 1,
                padding: 'valid'
            }));
            model.add(new primaryCapsLayer(32, 8));
            model.add(new digitCapsLayer(10, 16));

            this.model = model;

            this.optimizer = tf.train.adam();
        }

        loss(image, label) {
            return this.optimizer.minimize(() => {

                const vecLength = this.model.predict(image);
                const Tk = label;

                let maxL = tf.square(tf.maximum(0, 0.9 - vecLength));
                maxL = tf.reshape(maxL, [batchSize, -1]);

                let maxR = tf.square(tf.maximum(0., vecLength - 0.1));
                maxR = tf.reshape(maxR, [batchSize, -1]);

                const Lk = Tk * maxL + 0.5 * (1 - Tk) * maxR;
                const loss = tf.reduce_mean(tf.reduce_mean(Lk, 1))

                return loss;
            }, true);
        }

        accuracy(image, label) {
            return tf.tidy(() => {

                const vecLength = this.model.predict(image);
                label = tf.squeeze(tf.argmax(label, 1));
                
                const predictions = tf.squeeze(tf.argmax(vecLength, 1));
                const correct = tf.cast(tf.equal(label, predictions));
                const acc = tf.reduce_mean(correct) * 100
                
                return acc;
            })
        }
    }
};