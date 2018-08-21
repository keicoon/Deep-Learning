"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getExactlyOneTensor(xs) {
    let x;
    if (Array.isArray(xs)) {
        if (xs.length !== 1) {
            throw new Error(`Expected Tensor length to be 1; got ${xs.length}`);
        }
        x = xs[0];
    }
    else {
        x = xs;
    }
    return x;
}
exports.getExactlyOneTensor = getExactlyOneTensor;
function getExactlyOneShape(shapes) {
    if (Array.isArray(shapes) && Array.isArray(shapes[0])) {
        if (shapes.length === 1) {
            shapes = shapes;
            return shapes[0];
        }
        else {
            throw new Error(`Expected exactly 1 Shape; got ${shapes.length}`);
        }
    }
    else {
        return shapes;
    }
}
exports.getExactlyOneShape = getExactlyOneShape;
