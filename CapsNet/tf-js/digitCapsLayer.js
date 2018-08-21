module.exports = (tf) => {
    const CapsLayer = require('./capsLayer')(tf);

    return class DigitCapslayer extends CapsLayer {
        constructor(numOutputs, vecLen) {
            super(numOutputs, vecLen);
            this.supportsMasking = true;
        }

        computeOutputShape(inputShape) {
            // [128, 10, 16, 1]
            return [inputShape[0], 10, 16, 1];
        }

        call(inputs, kwargs) {
            let input = inputs;
            if (Array.isArray(input)) input = input[0];

            return tf.tidy(() => {
                const batchSize = input.shape[0];

                let uJI = input.reshape([batchSize, -1, 1, 8, 1]);

                let bIJ = tf.zeros([batchSize, 1152, 10, 1, 1], 'float32');
                let capsules = this.routing(uJI, bIJ);
                return tf.squeeze(capsules, 1);
            });
        }

        getClassName() {
            return 'DigitCapslayer';
        }
    }
}