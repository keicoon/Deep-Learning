module.exports = () => {

    const DQNHelper = require('./lib/dqn-tfjs');

    const params = DQNHelper.GetParameter();

    const firstState = [];
    const DQNSolver = new DQNHelper(params, firstState);

    const MAX_NUM_EPISODE = 100000;
    let numEpisode = 0;
    
    while (numEpisode < MAX_NUM_EPISODE) {
        const state = [];
        const actions = DQNSolver.decide(state);

        _.each(actions, action => {
            // step(action)
        })

        let reward = 0;
        let gameOver = false;

        DQNSolver.learn(reward, gameOver);
    }
}