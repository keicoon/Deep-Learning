
module.exports = (tf) => {

    function getVariable(shape) {
        return tf.randomNormal(shape);
    }

    function reduceSum(x, axis, keepDims = true) {
        return tf.sum(tf.square(x), axis, keepDims);
    }

    function matMul(a, b, transposeA = false, transposeB = false) {
        if (a.shape.length == b.shape.length
            && (transposeA ? a.shape[a.shape.length - 2] : a.shape[a.shape.length - 1]
                == transposeB ? b.shape[b.shape.length - 1] : b.shape[b.shape.length - 2])) {

            const shapeA = a.shape;
            const shapeB = b.shape;
            const arrA = a.dataSync();
            const arrB = b.dataSync();

            let arrMatMul2D = [];
            function matMul2D(boundA, boundB, shapeIdx) {
                if (shapeIdx < shapeA.length - 2) {
                    for (let i = 0; i < shapeA[shapeIdx]; i++) {
                        const countA = (boundA[1] - boundA[0]) / shapeA[shapeIdx];
                        const countB = (boundB[1] - boundB[0]) / shapeB[shapeIdx];
                        matMul2D(
                            [
                                boundA[0] + (countA * i),
                                boundA[0] + (countA * (i + 1))
                            ],
                            [
                                boundB[0] + (countB * i),
                                boundB[0] + (countB * (i + 1))
                            ], shapeIdx + 1);
                    }
                } else {
                    let matrix2dA = tf.tensor2d(arrA.slice(boundA[0], boundA[1]), [shapeA[shapeIdx], shapeA[shapeIdx + 1]]);
                    let matrix2dB = tf.tensor2d(arrB.slice(boundB[0], boundB[1]), [shapeB[shapeIdx], shapeB[shapeIdx + 1]]);
                    let arrResult = Array.from(matrix2dA.matMul(matrix2dB, transposeA, transposeB).dataSync())
                    Array.prototype.push.apply(arrMatMul2D, arrResult);
                }
            }

            function shape() {
                let arr = [];
                for (let i = 0; i < shapeA.length - 2; i++) {
                    arr.push(shapeA[i]);
                }
                arr.push(
                    transposeA ? a.shape[a.shape.length - 1] : a.shape[a.shape.length - 2],
                    transposeB ? b.shape[b.shape.length - 2] : b.shape[b.shape.length - 1]
                );
                return arr;
            }
            const arrShape = shape();
            matMul2D([0, arrA.length], [0, arrB.length], 0);

            return tf.tensor(arrMatMul2D, arrShape);
        } else {
            throw new Error('matMul shape not valid', 'a:', a.shape, 'b:', b.shape);
        }
    }

    return class CapsLayer extends tf.layers.Layer {
        constructor(numOutputs, vecLen) {
            super({});
            this.supportsMasking = true;

            this.numOutputs = numOutputs;
            this.vecLen = vecLen;
        }

        squash(vector) {
            const EPSILON = 1e-3;
            let vecSquaredNorm = reduceSum(vector, -2);
            let scalarFactor = vecSquaredNorm.div(vecSquaredNorm.add(1).mul(tf.sqrt(vecSquaredNorm.add(EPSILON))));
            let vecSquashed = scalarFactor.mul(vector);
            return vecSquashed;
        }

        routing(uJI, bIJ) {
            const batchSize = uJI.shape[0];

            let W = getVariable([1, 1152, 10, 8, 16]);

            uJI = uJI.tile([1, 1, 10, 1, 1]);

            W = W.tile([batchSize, 1, 1, 1, 1]);

            let uHat = matMul(W, uJI, true);

            let vJ;
            for (let i = 0; i < 3; i++) {
                // @NOTE: tf-js support only (rank - 1) value dimension. 
                let cIJ = bIJ.softmax(); // (2)

                let sJ = cIJ.mul(uHat);
                sJ = reduceSum(sJ, 1);

                vJ = this.squash(sJ);

                let vJTile = vJ.tile([1, 1152, 1, 1, 1]);

                let uV = matMul(uHat, vJTile, true);

                if (i < 2) {
                    bIJ += uV;
                }
            }

            return vJ;
        }
    }
}