import * as _ from 'lodash';

class Data {
    input: number[];
    label: number[];

    constructor(fn, range: number = 10) {
        const a = Math.floor(_.random(true) * range);
        const b = Math.floor(_.random(true) * range);
        this.input = [a, b];
        this.label = [fn(a, b)];
    }
}

export class DataSet {
    trainingSet: Data[];
    testingSet: Data[];
    train_idx: number = 0;
    test_idx: number = 0;
    
    fn: Function
    num_train: number
    num_test: number

    constructor(num_train: number = 1000, num_test: number = 1000, fn: Function) {
        this.fn = fn;
        this.num_train = num_train;
        this.num_test = num_test;
    }
    
    async loadData() {
        for (let i = 0; i < this.num_train; i++) {
            this.trainingSet[i] = new Data(this.fn);
        }
        for (let i = 0; i < this.num_test; i++) {
            this.testingSet[i] = new Data(this.fn);
        }
    }

    makeBatch(datas: Data[]) {
        let inputs: number[][];
        let labels: number[][];
        for (let i = 0; i < datas.length; i++) {
            inputs[i] = datas[i].input;
            labels[i] = datas[i].label;
        }
        return { inputs, labels };
    }
    // @TODO: Add `random batch`
    nextTrainBatch(num: number) {
        const len = this.trainingSet.length;
        if (this.train_idx < len) {
            const batchSet = this.trainingSet.splice(this.train_idx, this.train_idx + num);
            this.train_idx += num;
            return this.makeBatch(batchSet);
        } else {
            this.train_idx = 0;
            return this.nextTrainBatch(num);
        }
    }

    nextTestBatch(num: number) {
        const len = this.testingSet.length;
        if (this.train_idx < len) {
            const batchSet = this.testingSet.splice(this.train_idx, this.train_idx + num);
            this.train_idx += num;
            return this.makeBatch(batchSet);
        } else {
            this.train_idx = 0;
            return this.nextTestBatch(num);
        }
    }
}