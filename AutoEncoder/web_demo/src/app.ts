import * as express from "express";

class App {
    public app: express.Application;

    public static bootstrap (): App {
        return new App();
    }

    constructor() {
        this.app = express();
        this.app.get("/", (req: express.Request, res: express.Response, next: express.NextFunction) => {
            res.send("hello world")
        })
    }
}

export default App;


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