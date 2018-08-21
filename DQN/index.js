let UMG = require('UMG')
let viewport_widget = require('./lib/viewport-widget')

module.exports = (defer, reset) => {


    let elem = viewport_widget()
    defer(_ => {
        alive = false
        elem.destroy()
    })

    let inner = elem.add_child(
        UMG(Border,{
            BrushColor:{A:0.4}
        })
    )
    
    // require('TracePoint')(inner);
    // require('object-detection')(inner);
    require('snakeGame')(inner);

}