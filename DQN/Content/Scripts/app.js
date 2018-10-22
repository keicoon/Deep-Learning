/// <reference path="typings/ue.d.ts">/>
let _ = require('lodash')

let {defer,flush,reset} = require('./dqn/lib/defer')()

function main() {
    return require('./dqn')(defer,reset)
} 

try {
    module.exports = () => {
        process.nextTick(() => main());
        return flush
    }
}
catch (e) {
    // Context.CreateInspector(9229)
    require('bootstrap')('app.js')
}