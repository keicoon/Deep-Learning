#### A tfjs implementation of NAL(neural accumulator) and NALU(Neural Arithmetic Logic Units)

### Code snippet
```
import { NAC, NALU } from 'tfjs-nalu';

model.add(new NAC({
    units: num_output,
    inputShape: [num_input]
}));

model.add(new NALU({
    units: num_output,
    inputShape: [num_input],
    use_gating: true
}));
```

### Reference
- info from [paper](https://arxiv.org/pdf/1808.00508.pdf)
- keras from [github](https://github.com/titu1994/keras-neural-alu/)
- custom layer from [tfjs example](https://github.com/tensorflow/tfjs-layers/blob/v0.7.0/src/layers/core.ts)