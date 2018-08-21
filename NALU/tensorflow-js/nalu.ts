import * as tf from '@tensorflow/tfjs';
import * as util from './util';

export class NALU extends tf.layers.Layer {

    private outputDim: number;

    private W_Initializer: any;
    private W_Regularizer: any;
    private W_Constraint: any;

    private M_Initializer: any;
    private M_Regularizer: any;
    private M_Constraint: any;

    private gate_Initializer: any;
    private gate_Regularizer: any;
    private gate_Constraint: any;

    private W_hat: tf.LayerVariable;
    private M_hat: tf.LayerVariable;
    private G: tf.LayerVariable;

    private use_gating: boolean;
    private epsilon: number;

    constructor(config: any) {
        super(config);
        if (config.batchInputShape == null && config.inputShape == null &&
            config.inputDim != null) {
            // This logic is copied from Layer's constructor, since we can't
            // do exactly what the Python constructor does for Dense().
            let batchSize: number = null;
            if (config.batchSize != null) {
                batchSize = config.batchSize;
            }
            this.batchInputShape = [batchSize, config.inputDim];
        }
        this.outputDim = config.units;
        this.use_gating = config.use_gating;
        this.epsilon = 1e-7;

        this.W_Initializer = tf.initializers.glorotNormal({});
        this.W_Regularizer = tf.regularizers.l1l2({});
        this.W_Constraint = tf.constraints.unitNorm({});

        this.M_Initializer = tf.initializers.glorotNormal({});
        this.M_Regularizer = tf.regularizers.l1l2({});
        this.M_Constraint = tf.constraints.unitNorm({});

        this.gate_Initializer = tf.initializers.glorotNormal({});
        this.gate_Regularizer = tf.regularizers.l1l2({});
        this.gate_Constraint = tf.constraints.unitNorm({});

        this.supportsMasking = true;
        this.inputSpec = [{minNDim: 2}];
    }

    public build(inputShape: tf.Shape | tf.Shape[]): void {
        inputShape = util.getExactlyOneShape(inputShape);
        const inputLastDim = inputShape[inputShape.length - 1];

        this.W_hat = this.addWeight('W_hat', [inputLastDim, this.outputDim], null, this.W_Initializer,
            this.W_Regularizer, true, this.W_Constraint);
        this.M_hat = this.addWeight('M_hat', [inputLastDim, this.outputDim], null, this.M_Initializer,
            this.M_Regularizer, true, this.M_Constraint);
        if (this.use_gating) {
            this.G = this.addWeight('G', [inputLastDim, this.outputDim], null, this.gate_Initializer,
                this.gate_Regularizer, true, this.gate_Constraint);
        } else {
            this.G = null;
        }

        this.inputSpec = [{ minNDim: 2, axes: { [-1]: inputLastDim } }];
        this.built = true;
    }

    computeOutputShape(inputShape: tf.Shape | tf.Shape[]): tf.Shape | tf.Shape[] {
        inputShape = util.getExactlyOneShape(inputShape);
        const outputShape = inputShape.slice();
        outputShape[outputShape.length - 1] = this.outputDim;
        return outputShape;
    }

    call(inputs: tf.Tensor | tf.Tensor[], kwargs: any): tf.Tensor | tf.Tensor[] {
        return tf.tidy(() => {
            this.invokeCallHook(inputs, kwargs);
            // Dense layer accepts only a single input.
            const input = util.getExactlyOneTensor(inputs);

            const W = tf.mul(tf.tanh(this.W_hat.read()), tf.sigmoid(this.M_hat.read()));
            const m = tf.exp(tf.dot(tf.log(tf.add(tf.abs(input), this.epsilon)), W));
            const a = tf.dot(input, W);

            let outputs;
            if (this.use_gating) {
                const g = tf.sigmoid(tf.dot(input, this.G.read()));
                outputs = tf.add(tf.mul(g, a), tf.mul(tf.sub(1, g), m));
            } else {
                outputs = tf.add(a, m);
            }
            return outputs;
        });
    }

    getClassName() {
        return 'NALU';
    }
}