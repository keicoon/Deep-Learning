
let _ = require('lodash')
let UMG = require('UMG')
let compile = x => require('uclass')()(global, x)

module.exports = (elem) => {

    const snakeGameInstance = new SnakeGameInstance();

    const DQNHelper = require('./lib/dqn-tfjs');

    const params = DQNHelper.GetParameter(
        { width: WIDTH, height: HEIGHT, numAction: 4 }
    );
    const MAX_NUM_EPISODE = 100000;
    let numEpisode = 0;

    const DQNSolver = new DQNHelper(params, snakeGameInstance.state);

    const GAME_SPEED = 100;
    let act;
    act = () => {
        if (numEpisode < MAX_NUM_EPISODE) {
            const actions = DQNSolver.decide(snakeGameInstance.state);

            const action = actions[0]; snakeGameInstance.update(action);
            // actions.forEach(action => snakeGameInstance.update(action));

            let reward = snakeGameInstance.reward;
            let gameOver = snakeGameInstance.gameOver;

            DQNSolver.learn(reward, gameOver);

            if (gameOver) {
                numEpisode++;
                snakeGameInstance.reset();
            }

            setTimeout(act, GAME_SPEED);
        }
    }
    act();

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
            )
        ))
    }

    SetRenderWidget();
}

function initUE4Object(actor, color = { R: 255, G: 0, B: 0 }) {
    const mesh = StaticMesh.Load('/Engine/BasicShapes/Cube.Cube');
    const color_material = Material.Load('/Game/Color.Color');

    const material = KismetMaterialLibrary.CreateDynamicMaterialInstance(GWorld, color_material);
    material.SetVectorParameterValue('Color', color);

    actor.StaticMeshComponent.SetMobile();
    actor.StaticMeshComponent.SetStaticMesh(mesh);
    actor.StaticMeshComponent.SetMaterial(0, material);
}

const DELTA = 100, DELTA_Y = -450;
class SnakeGameObject {
    constructor(x = 0, y = 0, color) {
        this.x = x; this.y = y; this.setUEObject(color);
    }
    setPosition(x, y) {
        this.x = x; this.y = y;
        this.moveUEObject();
    }
    setUEObject(color) {
        this.ue = new StaticMeshActor(GWorld, { X: 0, Y: this.x * DELTA + DELTA_Y, Z: this.y * DELTA });
        initUE4Object(this.ue, color);
    }
    moveUEObject() {
        this.ue.SetActorLocation({ X: 0, Y: this.x * DELTA + DELTA_Y, Z: this.y * DELTA });
    }
}

const C_HEAD = { R: 0.5, G: 0, B: 1 };
const C_TAIL = { R: 0.7, G: 0, B: 1 };
const C_APPLE = { R: 1, G: 0, B: 0 };
const C_WALL = { R: 1, G: 1, B: 1 };

const E_ACTION = { Left: 0, Up: 1, Right: 2, Down: 3 };

class Snake extends SnakeGameObject {
    constructor() {
        super(3, 5, C_HEAD); // this object is HEAD
        this.tails = [];
    }

    reset() {
        this.setPosition(3, 5);
        this.tails.forEach(tail => tail.ue.DestroyActor())
        this.tails = [];
    }

    growup() { this.tails.push(new SnakeGameObject(this.x, this.y, C_TAIL)); }

    move(action) {
        if (this.prevAction) { // prevent moving back
            if (this.prevAction == E_ACTION.Left && action == E_ACTION.Right) action = E_ACTION.Left;
            if (this.prevAction == E_ACTION.Up && action == E_ACTION.Down) action = E_ACTION.Up;
            if (this.prevAction == E_ACTION.Right && action == E_ACTION.Left) action = E_ACTION.Right;
            if (this.prevAction == E_ACTION.Down && action == E_ACTION.Up) action = E_ACTION.Down;
        }

        let tail = this.tails.pop();
        if (tail) { // shift tail
            tail.setPosition(this.x, this.y);
            this.tails.unshift(tail);
        }
        switch (action) { // move head
            case E_ACTION.Left: this.setPosition(this.x - 1, this.y); break;
            case E_ACTION.Up: this.setPosition(this.x, this.y + 1); break;
            case E_ACTION.Right: this.setPosition(this.x + 1, this.y); break;
            case E_ACTION.Down: this.setPosition(this.x, this.y - 1); break;
            default: break;
        }

        this.prevAction = action;
    }
}

/* map head, tail, apple, wall
wwwwwwwwww
w        w
w        w
w        w
w        w
w h     aw
w t      w
w t      w
w        w
w        w
wwwwwwwwww
*/
const WIDTH = 10, HEIGHT = 10;
class SnakeGameInstance {
    constructor() {
        this.player = new Snake();
        this.apple = new SnakeGameObject(3, 3, C_APPLE);

        this.initMap();
        this.initGameInstance();
    }

    get state() { return this.map; }
    get reward() {
        const reward = this.gameOver ? -1 : this.getApple ? 2 : 1;
        this.getApple = false;
        return reward;
    }

    initMap() {
        this.map = new Array(WIDTH * HEIGHT);
        this.ueObjects = [];
        // setup wall
        for (let i = 0; i < this.map.length; i++) {
            const x = i % WIDTH, y = Math.floor(i / HEIGHT);
            if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) {
                let ueObject = new StaticMeshActor(GWorld, { X: 0, Y: x * DELTA + DELTA_Y, Z: y * DELTA });
                initUE4Object(ueObject, C_WALL);
                this.ueObjects.push(ueObject);
            }
        }
    }

    initGameInstance() {
        this.player.reset();

        this.render(); // first draw map

        const { appleX, appleY } = this.getAppleRandomPosition();
        this.apple.setPosition(appleX, appleY);
        this.getApple = false;
        this.gameOver = false;
    }

    reset() { this.initGameInstance() };

    update(action) {
        if (this.gameOver) return;

        this.player.move(action);

        if (this.checkCollision()) return;

        this.render();
    }

    render() {
        for (let i = 0; i < this.map.length; i++) {
            const x = i % WIDTH, y = Math.floor(i / HEIGHT);
            if (x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) { // wall
                this.map[i] = 1;
            } else if (x == this.player.x && y == this.player.y) { // head
                this.map[i] = 2;
            } else if (this.player.tails.some(tail => tail.x == x && tail.y == y)) { // tail
                this.map[i] = 3;
            } else if (x == this.apple.x && y == this.apple.y) { // apple
                this.map[i] = 4;
            } else {
                this.map[i] = 0;
            }
        }
    }

    checkCollision() {
        // check gameover
        const { x, y } = this.player;
        if (this.player.tails.some(tail => tail.x == x && tail.y == y) // self tail
            || x == 0 || x == WIDTH - 1 || y == 0 || y == HEIGHT - 1) { // wall
            return this.gameOver = true;
        }
        // check growup
        if (this.apple.x == x && this.apple.y == y) {
            this.getApple = true;
            const { appleX, appleY } = this.getAppleRandomPosition();
            this.apple.setPosition(appleX, appleY);

            this.player.growup();
        }
    }

    getAppleRandomPosition() {
        let random;
        while (true) {
            random = Math.floor(Math.random() * (WIDTH * HEIGHT));
            if (this.map[random] == 0) break;
        }
        return { appleX: random % WIDTH, appleY: Math.floor(random / HEIGHT) };
    }
};