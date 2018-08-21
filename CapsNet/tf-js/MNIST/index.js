
const tf = require('@tensorflow/tfjs');

const os = require('os');
// @NOTE: windows는 지원 안함..
if (!os.type().toLowerCase().includes('windows')) {
    require('@tensorflow/tfjs-node');
    // @NOTE: 엄청 느림.. nvidia-gpu라면 gpu-mode에서 실험을 추천함
    // require('@tensorflow/tfjs-node-gpu');
}

const CapsNet = require('../capsNet')(tf);
const MnistData = require('./data')(tf);

const BATCH_SIZE = 128
async function train() {
    let data = new MnistData();
    await data.loadData();

    let capsNet = new CapsNet();

    const iterNum = 100;
    for (let i = 0; i < iterNum; i++) {
        // test
        if (i % 10 === 0 && i > 0) {
            const testBatch = data.nextTestBatch(BATCH_SIZE);
            const acc = capsNet.accuracy(testBatch.image.reshape([BATCH_SIZE, 28, 28, 1]), testBatch.label);
            console.log(`[test] acc:${acc}`);
        } else {
            // train
            const batch = data.nextTrainBatch(BATCH_SIZE);
            const loss = capsNet.loss(batch.image.reshape([BATCH_SIZE, 28, 28, 1]), batch.label);
            console.log(`[train] step: ${i + 1} loss: ${loss}`);
        }
    }
}

train()