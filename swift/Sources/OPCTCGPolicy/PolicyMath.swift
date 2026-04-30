import Foundation

@inline(__always)
func leakyRelu(_ x: Float, alpha: Float = 0.1) -> Float {
    x >= 0 ? x : alpha * x
}

/// ``out[o] = sum_i W[o,i] * x[i] + b[o]`` avec ``W`` ligne-major ``[outDim × inDim]``.
func linearMatVec(
    x: [Float],
    weight: [Float],
    bias: [Float],
    outDim: Int,
    inDim: Int
) -> [Float] {
    precondition(weight.count == outDim * inDim)
    precondition(bias.count == outDim)
    precondition(x.count == inDim)
    var y = [Float](repeating: 0, count: outDim)
    for o in 0..<outDim {
        var s = bias[o]
        let row = o * inDim
        for i in 0..<inDim {
            s += weight[row + i] * x[i]
        }
        y[o] = s
    }
    return y
}

/// LayerNorm sur le dernier axe (vecteur longueur ``n``), epsilon ``1e-5`` comme PyTorch / MLX.
func layerNormVec(x: [Float], gamma: [Float], beta: [Float], eps: Float = 1e-5) -> [Float] {
    let n = x.count
    precondition(gamma.count == n && beta.count == n)
    var mean: Float = 0
    for v in x { mean += v }
    mean /= Float(n)
    var varAcc: Float = 0
    for v in x {
        let d = v - mean
        varAcc += d * d
    }
    varAcc /= Float(n)
    let inv = 1 / sqrt(varAcc + eps)
    var y = [Float](repeating: 0, count: n)
    for i in 0..<n {
        y[i] = (x[i] - mean) * inv * gamma[i] + beta[i]
    }
    return y
}
