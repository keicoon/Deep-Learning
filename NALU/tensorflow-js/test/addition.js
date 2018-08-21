"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs");
const _ = require("lodash");
const index_1 = require("../index");
const model = tf.sequential();
model.add(new index_1.NAC({
    units: 1,
    inputShape: [2]
}));
const optimizer = tf.train.adam();
class Data {
    constructor(range = 10) {
        const a = Math.floor(_.random(true) * range);
        const b = Math.floor(_.random(true) * range);
        this.input = [a, b];
        this.label = [this.fn(a, b)];
    }
    fn(a, b) {
        return a + b;
    }
}
class BuildData {
    constructor() {
        this.train_idx = 0;
        this.test_idx = 0;
    }
    loadData(num_train = 1000, num_test = 1000) {
        return __awaiter(this, void 0, void 0, function* () {
            for (let i = 0; i < num_train; i++) {
                this.trainingSet[i] = new Data();
            }
            for (let i = 0; i < num_test; i++) {
                this.testingSet[i] = new Data();
            }
        });
    }
    makeBatch(datas) {
        let inputs;
        let labels;
        for (let i = 0; i < datas.length; i++) {
            inputs[i] = datas[i].input;
            labels[i] = datas[i].label;
        }
        return { inputs, labels };
    }
    nextTrainBatch(num) {
        const len = this.trainingSet.length;
        if (this.train_idx < len) {
            const batchSet = this.trainingSet.splice(this.train_idx, this.train_idx + num);
            this.train_idx += num;
            return this.makeBatch(batchSet);
        }
        else {
            // @TODO:
            this.train_idx = 0;
            return this.nextTrainBatch(num);
        }
    }
    nextTestBatch(num) {
        const len = this.testingSet.length;
        if (this.train_idx < len) {
            const batchSet = this.testingSet.splice(this.train_idx, this.train_idx + num);
            this.train_idx += num;
            return this.makeBatch(batchSet);
        }
        else {
            // @TODO:
            this.train_idx = 0;
            return this.nextTestBatch(num);
        }
    }
}
let data = new BuildData();
function GetLoss(input, label) {
    return optimizer.minimize(() => {
        const y = model.predict(input);
        const loss = tf.mean(tf.square(tf.sub(y, label)), -1);
        return tf.sum(loss);
    }, true);
}
function GetAccuracy(input, label) {
    return optimizer.minimize(() => {
        const y = model.predict(input);
        const acc = tf.mean(tf.square(tf.sub(y, label)), -1);
        return tf.mean(acc);
    }, true);
}
function train() {
    return __awaiter(this, void 0, void 0, function* () {
        yield data.loadData();
        const iterNum = 100;
        for (let i = 0; i < iterNum; i++) {
            if (i % 10 === 0 && i > 0) { // test
                const { inputs, labels } = data.nextTestBatch(128);
                const acc = GetAccuracy(inputs, labels);
                console.log(`[test] acc:${acc}`);
            }
            else { // train
                const { inputs, labels } = data.nextTrainBatch(128);
                const loss = GetLoss(inputs, labels);
                console.log(`[train] step: ${i + 1} loss: ${loss}`);
            }
        }
    });
}
train();
