'use strict';

const tf = require('./lib/fancyLoadNPM')('@tensorflow/tfjs', global);

class Util {
    static mse(predictions, targets, mask) {
        return tf.mul(predictions.sub(targets.expandDims(1)).square(), mask.asType('float32')).mean();
    }
}

class DQNModel {
    constructor(params) {
        this.parameters = params;

        const model = tf.sequential();
        // (10 x 10 x 1)
        model.add(tf.layers.conv2d({
            inputShape: [params.width, params.height, params.stackFrames],
            filters: 5,
            kernelSize: 3,
            strides: 1,
            activation: 'relu'
        }));
        model.add(tf.layers.flatten());
        // (8 x 8 x 5)
        model.add(tf.layers.dense({
            units: 32,
            activation: 'relu',
            inputDim: 320
        }));
        // (32)
        model.add(tf.layers.dense({
            units: params.numAction,
            activation: 'linear',
            inputDim: 32
        }));

        /* This is model of DeepMind.
        model.add(tf.layers.conv2d({
            inputShape: [params.width, params.height, params.stackFrames],
            filters: 32,
            kernelSize: 8,
            strides: 4,
            activation: 'relu'
        }));

        model.add(tf.layers.conv2d({
            inputShape: [20, 20, 32],
            filters: 64,
            kernelSize: 4,
            strides: 2,
            activation: 'relu'
        }));

        model.add(tf.layers.conv2d({
            inputShape: [9, 9, 64],
            filters: 64,
            kernelSize: 3,
            strides: 1,
            activation: 'relu'
        }));

        model.add(tf.layers.dense({
            units: 512,
            activation: 'relu',
            inputDim: 3136 // (7 * 7 * 64)
        }));

        model.add(tf.layers.dense({
            units: params.numAction,
            activation: 'linear',
            inputDim: 512
        }));
        */

        this._model = model;
    }

    getTargets(batchR, batchNextS, batchDone) {
        return tf.tidy(() => {
            const maxQ = this.predict(batchNextS.reshape(
                [-1, this.parameters.width, this.parameters.height, this.parameters.stackFrames]
            )).max(1);
            const targets = batchR.add(maxQ.mul(tf.scalar(this.parameters.discount)).mul(batchDone));
            return targets;
        });
    }

    getLoss(replay, targetModel, modelVars, optimizer) {
        const arrayPrevS = [];
        const arrayA = [];
        const arrayR = [];
        const arrayNextS = [];
        const arrayDone = [];

        for (let i = 0; i < this.parameters.minibatchSize; i++) {
            const exp = replay[Math.floor(Math.random() * replay.length)];
            arrayPrevS.push(exp.prevS);
            arrayA.push(exp.action);
            arrayNextS.push(exp.nextS);
            arrayR.push(exp.reward);
            arrayDone.push(exp.done ? 0 : 1);
        }

        const batchPrevS = tf.tensor2d(arrayPrevS);
        const batchA = tf.tensor1d(arrayA, 'int32');
        const batchR = tf.tensor1d(arrayR);
        const batchNextS = tf.tensor2d(arrayNextS);
        const batchDone = tf.tensor1d(arrayDone);

        const predMask = tf.oneHot(batchA, this.parameters.numAction);

        const targets = targetModel.getTargets(batchR, batchNextS, batchDone);

        const loss = optimizer.minimize(() => {
            const x = tf.variable(batchPrevS);
            const predictions = this.predict(x.reshape(
                [-1, this.parameters.width, this.parameters.height, this.parameters.stackFrames]
            ));
            const re = Util.mse(predictions, targets, predMask);
            x.dispose();

            return re;
        }, true, modelVars);

        targets.dispose();

        batchPrevS.dispose();
        batchA.dispose();
        batchR.dispose();
        batchNextS.dispose();
        batchDone.dispose();

        predMask.dispose();

        return loss;
    }

    predict(...p) { return this._model.predict.apply(this._model, p); }
    get weights() { return this._model.weights; }
}

class DQNHelper {
    constructor(params, state) {
        this.parameters = params

        this.replay = [];
        this.states = [state];
        this.scores = [];

        this.epFrames = 0;
        this.totalFrames = 0;
        this.lossc = -1;
        this.savedDecideStatus = {};
        /// Intialize
        {
            this.optimizer = tf.train.adam(params.learningRate);
            this.model = new DQNModel(params);
            this.targetModel = new DQNModel(params);
            this.syncModelWeights(this.model, this.targetModel);
            this.setModelVars(this.model);
            this.callbackUpdateLoss = () => { };
            this.callbackUpdateScore = () => { };
        }
        console.log("training...");
    }

    makeStackObs() {
        let arrays = [];
        for (let i = 0; i < this.parameters.stackFrames; i++) {
            arrays.push(this.states[Math.max(0, this.states.length - 1 - i)]);
        }
        return Array.prototype.concat.apply([], arrays);
    }

    decide(state) { // state is [] , actions is []
        const observation = this.makeStackObs();
        let act = Math.floor(Math.random() * this.parameters.numAction);
        const obsTensor = tf.tensor2d([observation]);
        const vals = this.model.predict(obsTensor.reshape(
            [-1, this.parameters.width, this.parameters.height, this.parameters.stackFrames]
        ));
        obsTensor.dispose();

        const a = Math.min(1, this.totalFrames / this.parameters.finExpFrame);
        if (this.replay.length >= this.parameters.replayStartSize && Math.random() > a * this.parameters.finExp + (1 - a) * this.parameters.initExp) {
            const maxAct = vals.argMax(1);
            act = maxAct.dataSync();
            maxAct.dispose();
        }

        const normVals = tf.softmax(vals);
        this.savedDecideStatus = { vals: vals.dataSync(), normVals: normVals.dataSync(), act, observation };
        vals.dispose();
        normVals.dispose();

        let actions = [];
        for (let i = 0; i < this.parameters.actionRepeat; i++) {
            actions.push(act)
        };

        this.states.push(state);

        return actions;
    }

    learn(reward, gameOver = false) { // reawrd is float, gameOver is boolean
        const nextS = this.makeStackObs();

        const { observation, act } = this.savedDecideStatus;
        this.replay.push({
            prevS: observation,
            action: act,
            reward: reward,
            nextS: nextS,
            done: gameOver
        });

        if (this.replay.length > this.parameters.replayMemorySize) {
            this.replay = this.replay.slice(this.replay.length - this.parameters.replayMemorySize);
        }

        if (this.replay.length >= this.parameters.replayStartSize) {
            const loss = this.model.getLoss(this.replay, this.targetModel, this.modelVars, this.optimizer);

            if (gameOver) {
                this.lossc = loss.dataSync()[0];
                this.callbackUpdateLoss(this.epFrames);
                // console.log("loss: " + this.lossc);
            }
            loss.dispose();
        }

        this.epFrames++;
        this.totalFrames++;

        if (this.totalFrames % this.parameters.targetUpdateFreq === 0) {
            this.syncModelWeights(this.model, this.targetModel);
            console.log("frame: " + this.totalFrames);
            console.log("replay buffer: " + this.replay.length);
            console.log("numTensors: " + tf.memory().numTensors);
        }

        if (gameOver) {
            this.callbackUpdateScore(this.epFrames);
            this.scores.push(this.epFrames);
            this.epFrames = 0;
            const lastState = this.states[this.states.length - 1];
            this.states = [lastState];
        }
    }

    syncModelWeights(model, targetModel) {
        console.log("sync target model weights");
        for (let i = 0; i < model.weights.length; i++) {
            targetModel.weights[i].val.assign(model.weights[i].val);
        }
    }

    setModelVars(model) {
        this.modelVars = [];
        for (let i = 0; i < model.weights.length; i++) {
            this.modelVars.push(model.weights[i].val);
        }
    }

    get loss() { return this.lossc; }

    static GetParameter(params) {
        const defaultParams = {
            minibatchSize: 32,
            replayMemorySize: 10000,
            stackFrames: 2,
            targetUpdateFreq: 100,
            discount: 0.99,
            actionRepeat: 4,
            learningRate: 0.001,
            initExp: 1.0,
            finExp: 0.1,
            finExpFrame: 10000,
            replayStartSize: 100,
            hiddenLayers: [64, 64],
            activation: 'elu',
            width: 20,
            height: 20,
            numAction: 3
        };

        let parameters = {};
        for (const key in defaultParams) {
            const value = defaultParams[key];
            parameters[key] = params[key] || value
        }
        return parameters;
    }

}

module.exports = DQNHelper;