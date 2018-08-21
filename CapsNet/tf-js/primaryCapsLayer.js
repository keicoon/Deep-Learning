module.exports = (tf) => {
    const CapsLayer = require('./capsLayer')(tf);

    return class PrimaryCapsLayer extends CapsLayer {
        constructor(numOutputs, vecLen) {
            super(numOutputs, vecLen);
            this.supportsMasking = true;

            this.conv = tf.layers.conv2d({
                inputShape: [20, 20, 256],
                filters: numOutputs * vecLen,
                kernelSize: 9,
                strides: 2,
                padding: 'valid'
            });
        }

        computeOutputShape(inputShape) {
            // const dim = this.numOutputs * Math.pow(Math.floor(((inputShape[1] - this.kernelSize) / this.strides) + 1), 2)
            // [128, 1152, 8 1]
            return [inputShape[0], 1152, 8, 1];
        }

        call(inputs, kwargs) {
            let input = inputs;
            if (Array.isArray(input)) input = input[0];
            
            return tf.tidy(() => {
                const batchSize = input.shape[0];

                let capsules = this.conv.apply(input);
                capsules = capsules.reshape([batchSize, -1, 8, 1]);
                return this.squash(capsules);
            });
        }

        getClassName() {
            return 'PrimaryCapsLayer';
        }
    }
}