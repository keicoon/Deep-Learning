"use strict";
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (Object.hasOwnProperty.call(mod, k)) result[k] = mod[k];
    result["default"] = mod;
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
const express = __importStar(require("express"));
class App {
    static bootstrap() {
        return new App();
    }
    constructor() {
        this.app = express();
        this.app.get("/", (req, res, next) => {
            res.send("hello world");
        });
    }
}
exports.default = App;
// # example of tfjs-converter
// [ https://github.com/tensorflow/tfjs-converter, https://js.tensorflow.org/tutorials/model-save-load.html ]
// import * as tf from '@tensorflow/tfjs';
// import {loadFrozenModel} from '@tensorflow/tfjs-converter';
// const MODEL_URL = 'https://.../mobilenet/web_model.pb';
// const WEIGHTS_URL = 'https://.../mobilenet/weights_manifest.json';
// const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL);
// const cat = document.getElementById('cat');
// model.execute({input: tf.fromPixels(cat)});
// #ref : https://developer.mozilla.org/en-US/docs/Web/API/WindowOrWorkerGlobalScope/fetch
// const model = await loadFrozenModel(MODEL_URL, WEIGHTS_URL, {credentials: 'include'});
//# sourceMappingURL=app.js.map