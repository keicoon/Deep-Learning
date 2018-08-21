# DQN in ue4
### Model
```
DQN Model
Layer	Input	        FilterSize  Stride  Filter 갯수   Activation  Output
conv1	84 x 84 x 4     8 x 8       4	    32	        ReLU        20 x 20 x 32
conv2	20 x 20 x 32	4 x 4       2	    64	        ReLU        9 * 9 * 64
conv2	9 x 9 x 64      3 x 3       1	    64	        ReLU        7 x 7 x 64
fc1     7 x 7 x 64                          512	        ReLU        512
fc2     512                                 2	        Linear      2
```
### Experiment
1. TracePoint
    - Train to move position A to B in 2D.
    - ![](https://github.com/keicoon/DQN/blob/master/capture/TracePoint.PNG)
2. DetectionObject
    - Train to detect dangerous object in variation.
    - ![](https://github.com/keicoon/DQN/blob/master/capture/object-detection.png)
3. SnakeGame
    - Train snake to survive smartly more than normal algorithm.
    - ![](https://github.com/keicoon/DQN/blob/master/capture/snakeGame.png)
4. HideAndSeek
    - Train AI in HideAndSeek.

### DQN Lib
- Using `reinforce-js` casue some problems.  
- I replace that to `dqn-tfjs` made in tsfs.  

### Dependencies
1. [unreal.js](https://github.com/ncsoft/Unreal.js)
2. [reinforce-js](https://github.com/mvrahden/reinforce-js)
3. [@tensorflow-js](https://github.com/tensorflow/tfjs)
