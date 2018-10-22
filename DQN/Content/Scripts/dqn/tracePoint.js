
let _ = require('lodash')
let UMG = require('UMG')
let compile = x => require('uclass')()(global, x)

module.exports = (elem) => {

    const { DQNSolver, DQNOpt, DQNEnv } = require('./lib/fancyLoadNpm')('reinforce-js', global);

    const width = 5;
    const height = 5;
    const numberOfStates = 4;
    const numberOfActions = 4;
    const env = new DQNEnv(width, height, numberOfStates, numberOfActions);

    const opt = new DQNOpt();
    opt.setTrainingMode(true);
    opt.setNumberOfHiddenUnits([100]);  // mind the array here, currently only one layer supported! Preparation for DNN in progress...
    opt.setEpsilonDecay(1.0, 0.1, 1e6);
    opt.setEpsilon(0.05);
    opt.setGamma(0.9);
    opt.setAlpha(0.005);
    opt.setLossClipping(true);
    opt.setLossClamp(1.0);
    opt.setRewardClipping(true);
    opt.setRewardClamp(1.0);
    opt.setExperienceSize(1e6);
    opt.setReplayInterval(5);
    opt.setReplaySteps(5);

    const dqnSolver = new DQNSolver(env, opt);

    const state = [5, 5, 0, 0];

    const UpdateState = (action) => {
        const moveStep = 0.1;
        if (action == 0) { // left
            state[2] -= moveStep;
        } else if (action == 1) { // right
            state[2] += moveStep;
        } else if (action == 2) { // up
            state[3] += moveStep;
        } else if (action == 3) { // down
            state[3] -= moveStep;
        }

        state[2] = _.clamp(state[2], 0, 5);
        state[3] = _.clamp(state[3], 0, 5);
    }

    const GetActionReward = (action) => {
        const DEFAULT_REWARD = 1;

        let reword = 0;

        // const dx = state[0] - state[2];
        // const dy = state[1] - state[3];

        // if (dx > 2.5) {
        //     if (action == 1) reword = DEFAULT_REWARD;
        // }
        // if (dx < -2.5) {
        //     if (action == 0) reword = DEFAULT_REWARD;
        // }
        // if (dy > 2.5) {
        //     if (action == 2) reword = DEFAULT_REWARD;
        // }
        // if (dy < -2.5) {
        //     if (action == 3) reword = DEFAULT_REWARD;
        // }

        return reword;
    }

    const GetStateReward = () => {
        let reword = 0;

        const dx = Math.abs(state[0] - state[2]);
        const dy = Math.abs(state[1] - state[3]);
        const dist = Math.sqrt(dx * dx + dy * dy);

        reword = -1 * (dist) * (dist) * 0.01;

        return reword;
    }

    const ToFixedString = (floatArray) => {
        let str = '';
        floatArray.forEach(v => {
            str += v.toFixed(2) + ' ';
        })
        return str;
    }

    let reword = 0;
    setInterval(() => {
        const action = dqnSolver.decide(state);

        UpdateState(action);

        const rewardAction = GetActionReward(action);
        const rewardState = GetStateReward();
        reword = rewardAction + rewardState;

        lossSet(reword);

        dqnSolver.learn(reword);
    }, 33);

    let lossSet = () => { };
    let brushAsset = new SlateBrushAsset
    class RenderWidget extends JavascriptWidget {
        OnPaint(_context) {
            let context = PaintContext.C(_context)

            const OFFSET = 200;
            const SCALE = 40;

            context.DrawBox(
                { X: state[2] * SCALE + OFFSET, Y: state[3] * SCALE },
                { X: 10, Y: 10 },
                brushAsset,
                { R: 0, G: 1, B: 0, A: 1 }
            );
            context.DrawBox(
                { X: state[0] * SCALE + OFFSET, Y: state[1] * SCALE },
                { X: 10, Y: 10 },
                brushAsset,
                { R: 1, G: 0, B: 0, A: 1 }
            );
        }
    }

    class Graph extends JavascriptWidget {
        ctor() {
            // exponential moving average of (max)
            this.ema = 1
            this.data = []
        }
        OnPaint(context) {
            let max = _.max(this.data) || 0
            this.ema = max = this.ema * 0.95 + max * 0.05
            const sx = 1
            const sy = 100
            function p(x, y) {
                return { X: x * sx + sx, Y: -y / (max + 1e-3) * sy + sy + 100 }
            }
            let py
            this.data.forEach((y, x) => {
                if (py != undefined) {
                    PaintContext.C(context).DrawLine(
                        p(x - 1, py),
                        p(x, y),
                        { R: 1, G: 0.7, B: 0, A: 1 },
                        true)
                }

                py = y
            })
        }
    }

    function SetRenderWidget() {
        let widget = elem.add_child(UMG.div({},
            UMG.text({ $link: elem => elem.TextDelegate = () => `CurrPosition (${ToFixedString([state[2], state[3]])})` }),
            UMG.text({ $link: elem => elem.TextDelegate = () => `WishPosition (${ToFixedString([state[0], state[1]])})` }),
            UMG.text({ $link: elem => elem.TextDelegate = () => `Reward (${ToFixedString([reword])})` }),
            UMG(SizeBox, { WidthOverride: 400, HeightOverride: 200 },
                UMG(Border, { BrushColor: { A: 0.4 } },
                    UMG(compile(Graph), {
                        $link: elem => {
                            let data = []
                            lossSet = sample => {
                                data.push(sample)
                                if (data.length > 800) {
                                    data.shift()
                                }
                                elem.data = data
                            }
                        }
                    })
                )
            ),
            UMG(compile(RenderWidget), {})
        ))
    }

    SetRenderWidget();
}