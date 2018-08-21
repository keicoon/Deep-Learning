import * as tf from '@tensorflow/tfjs';

export function getExactlyOneTensor(xs: tf.Tensor | tf.Tensor[]): tf.Tensor {
    let x: tf.Tensor;
    if (Array.isArray(xs)) {
        if (xs.length !== 1) {
            throw new Error(`Expected Tensor length to be 1; got ${xs.length}`);
        }
        x = xs[0];
    } else {
        x = xs as tf.Tensor;
    }
    return x;
}

export function getExactlyOneShape(shapes: tf.Shape | tf.Shape[]): tf.Shape {
    if (Array.isArray(shapes) && Array.isArray(shapes[0])) {
        if (shapes.length === 1) {
            shapes = shapes as tf.Shape[];
            return shapes[0];
        } else {
            throw new Error(`Expected exactly 1 Shape; got ${shapes.length}`);
        }
    } else {
        return shapes as tf.Shape;
    }
}