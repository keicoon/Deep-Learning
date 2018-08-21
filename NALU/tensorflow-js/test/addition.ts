import * as tf from '@tensorflow/tfjs';
import { DataSet } as from './dataSet';
import { NAC } from '../index';

const data = new DataSet(1028, 1028, (a, b) => (a + b));
await data.loadData();

class Experiment {
    model: any
    optimizer: any
    num_iter: number

    constructor(model, optimizer) {
        this.num_iter = 100;
        this.model = model;
        this.optimizer = optimizer;
    }

    getLoss(input, label) {
        return this.optimizer.minimize(() => {
            const y = <tf.Tensor>this.model.predict(input);
            const loss = tf.mean(tf.square(tf.sub(y, label)), -1);
            return tf.sum(loss);
        }, true);
    }

    getAccuracy(input, label) {
        return this.optimizer.minimize(() => {
            const y = <tf.Tensor>this.model.predict(input);
            const acc = tf.mean(tf.square(tf.sub(y, label)), -1);
            return tf.mean(acc);
        }, true);
    }

    public train() {
        for (let i = 0; i < this.num_iter; i++) {
            if (i % 10 === 0 && i > 0) { // test
                const { inputs, labels } = data.nextTestBatch(128);
                const acc = this.getAccuracy(inputs, labels);
                console.log(`[test] acc:${acc}`);
            } else { // train
                const { inputs, labels } = data.nextTrainBatch(128);
                const loss = this.getLoss(inputs, labels);
                console.log(`[train] step: ${i + 1} loss: ${loss}`);
            }
        }
    }
}

const nac = () => {
    const model = tf.sequential();

    model.add(new NAC({
        units: 1,
        inputShape: [2]
    }));

    const optimizer = tf.train.adam();
    console.log(`NAC`);
    const ex = new Experiment(model, optimizer);
    ex.train();
}

const relu = () => {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        units: 1,
        activation: 'relu',
        inputDim: 2
    }));

    const optimizer = tf.train.adam();
    console.log(`RELU`);
    const ex = new Experiment(model, optimizer);
    ex.train();
}
// @NOTE: main
nac();
relu();