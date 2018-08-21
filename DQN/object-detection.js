
let _ = require('lodash')
let UMG = require('UMG')
let compile = x => require('uclass')()(global, x)

module.exports = (elem) => {

    const DQNHelper = require('./lib/dqn-tfjs');

    const params = DQNHelper.GetParameter({
        width: Math.floor(WIDTH), height: Math.floor(HEIGHT), numAction: 2
    });

    const WIDTH = 10.0, HEIGHT = 10.0;
    const E_ACTION = { DETECTION: 0, NONE: 1 };

    function getDistance2D(a, b) {
        return Math.sqrt((a.X - b.X) * (a.X - b.X) + (a.Y - b.Y) * (a.Y - b.Y));
    }

    let playerPosition = { X: 5.0, Y: 5.0 };
    let targetPosition = { X: 2.0, Y: 2.0 };
    const RANGE_SENSOR = 2.5;
    const MAX_RANGE = Math.sqrt(WIDTH * WIDTH + HEIGHT * HEIGHT);

    const UpdateState = (action) => {
        targetPosition.X = _.random(0, WIDTH);
        targetPosition.Y = _.random(0, HEIGHT);
    }

    const GetState = () => {
        let state = [playerPosition.X / WIDTH, playerPosition.Y / HEIGHT];
        // sensor
        state.push(getDistance2D(playerPosition, targetPosition) / MAX_RANGE);

        return state;
    }

    const GetReward = (action) => {
        let reward = -1;
        let gameOver = false;

        const dist = getDistance2D(playerPosition, targetPosition);
        const isRanged = (dist <= RANGE_SENSOR);

        if ((action == E_ACTION.DETECTION && isRanged) || action == E_ACTION.NONE && !isRanged) {
            reward = 1;
        } else {
            if (--life < 1) {
                gameOver = true;
                life = 1;
            }
        }

        return { reward, gameOver };
    }

    const DQNSolver = new DQNHelper(params, GetState());

    let action = 0;
    let life = 1;
    setInterval(() => {
        const actions = DQNSolver.decide(GetState());

        action = _.last(actions); // don't repeat

        UpdateState(action);

        const { reward, gameOver } = GetReward(action);

        DQNSolver.learn(reward, gameOver);
    }, 200);

    let brushAsset = new SlateBrushAsset
    const THICKNESS = 10;
    const NUM_CIRCLE = 36;
    const OFFSET = 200;
    const SCALE = 40;
    class RenderWidget extends JavascriptWidget {
        OnPaint(_context) {
            let context = PaintContext.C(_context)

            context.DrawBox(
                { X: playerPosition.X * SCALE + OFFSET, Y: playerPosition.Y * SCALE },
                { X: THICKNESS, Y: THICKNESS },
                brushAsset,
                { R: 0, G: 0, B: 1, A: 1 }
            );

            context.DrawBox(
                { X: targetPosition.X * SCALE + OFFSET, Y: targetPosition.Y * SCALE },
                { X: THICKNESS, Y: THICKNESS },
                brushAsset,
                (action == E_ACTION.DETECTION) ? { R: 1, G: 0, B: 0, A: 1 } : { R: 0, G: 1, B: 0, A: 1 }
            );

            let circles = [];
            for (let i = 0; i <= NUM_CIRCLE; i++) {
                circles.push({
                    X: (playerPosition.X + Math.cos(Math.PI * i / (NUM_CIRCLE / 2)) * RANGE_SENSOR) * SCALE + OFFSET,
                    Y: (playerPosition.Y + Math.sin(Math.PI * i / (NUM_CIRCLE / 2)) * RANGE_SENSOR) * SCALE,
                })
            }
            context.DrawLines(
                circles,
                { R: 0.8, G: 0.4, B: 0.4, A: 1 },
                true
            )
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
            UMG.text({ $link: elem => elem.TextDelegate = () => `loss: ${DQNSolver.loss}` }),
            UMG(SizeBox, { WidthOverride: 400, HeightOverride: 200 },
                UMG(Border, { BrushColor: { A: 0.4 } },
                    UMG(compile(Graph), {
                        $link: elem => {
                            let data = []
                            DQNSolver.callbackUpdateLoss = sample => {
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
            UMG.text({ $link: elem => elem.TextDelegate = () => `epNum: ${DQNSolver.scores.length}` }),
            UMG(SizeBox, { WidthOverride: 400, HeightOverride: 200 },
                UMG(Border, { BrushColor: { A: 0.4 } },
                    UMG(compile(Graph), {
                        $link: elem => {
                            let data = []
                            DQNSolver.callbackUpdateScore = sample => {
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