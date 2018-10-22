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
### SnakeGame
- Train snake to survive smartly more than normal algorithm.
- ![](https://github.com/keicoon/Deep-Learning/blob/master/DQN/capture/snakeGame.png)

### DQN Lib
- I use `dqn-tfjs` made in tsfs.  
- If you want to source just, visit gitlab-repo(https://gitlab.com/keicoon15/uneral.js-rl)

### Dependencies
1. [unreal.js](https://github.com/ncsoft/Unreal.js)
2. [@tensorflow-js](https://github.com/tensorflow/tfjs)