local Math = require(script.Parent.MathOps)
local Act = require(script.Parent.Activations)
local W1 = require(script.Parent.Parent.Weights.W1)
local B1 = require(script.Parent.Parent.Weights.B1)
local W2 = require(script.Parent.Parent.Weights.W2)
local B2 = require(script.Parent.Parent.Weights.B2)

return {
    predict = function(x)
        local h1 = Math.dot(W1, x)

        for i = 1, #h1 do
            h1[i] += B1[i]
        end
        h1 = Act.relu(h1)

        local out = Math.dot(W2, h1)
        for i = 1, #out do
            out[i] += B2[i]
        end
        
        return Act.softmax(out)
    end
}