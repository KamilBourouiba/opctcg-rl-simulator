import Foundation
import OPTrainingCore

public enum SwiftComputeRuntime {
    private static var configuredThreads: Int = {
        let cores = ProcessInfo.processInfo.activeProcessorCount
        return max(1, cores)
    }()

    public static func setThreads(_ n: Int) {
        configuredThreads = max(1, n)
    }

    public static func autoConfigure() {
        configuredThreads = max(1, ProcessInfo.processInfo.activeProcessorCount)
    }

    public static var threads: Int { configuredThreads }
}

/// Politique linéaire + baseline scalaire (régression sur ``obs``).
public struct LinearActorCritic {
    public let obsDim: Int
    public let nAct: Int
    /// Lignes ``n_act × obs_dim`` (row-major).
    public var wPi: [Float]
    public var bPi: [Float]
    public var wVal: [Float]
    public var bVal: Float

    public init(obsDim: Int, nAct: Int, rng: inout SplitMix64) {
        self.obsDim = obsDim
        self.nAct = nAct
        let dw = obsDim * nAct
        self.wPi = (0..<dw).map { _ in Float(rng.nextDouble(in: -0.02...0.02)) }
        self.bPi = (0..<nAct).map { _ in Float(rng.nextDouble(in: -0.02...0.02)) }
        self.wVal = (0..<obsDim).map { _ in Float(rng.nextDouble(in: -0.02...0.02)) }
        self.bVal = Float(rng.nextDouble(in: -0.05...0.05))
    }

    public func logits(obs: [Float], mask: [Bool]) -> [Float] {
        precondition(obs.count == obsDim)
        precondition(mask.count == nAct)
        var out = [Float](repeating: 0, count: nAct)
        let useParallel = SwiftComputeRuntime.threads > 1 && nAct >= 32
        if useParallel {
            out.withUnsafeMutableBufferPointer { outBuf in
                DispatchQueue.concurrentPerform(iterations: nAct) { a in
                    var s = bPi[a]
                    let row = a * obsDim
                    for k in 0..<obsDim {
                        s += wPi[row + k] * obs[k]
                    }
                    outBuf[a] = mask[a] ? s : -1e9
                }
            }
        } else {
            for a in 0..<nAct {
                var s = bPi[a]
                let row = a * obsDim
                for k in 0..<obsDim {
                    s += wPi[row + k] * obs[k]
                }
                out[a] = mask[a] ? s : -1e9
            }
        }
        return out
    }

    public func value(obs: [Float]) -> Float {
        var v = bVal
        for k in 0..<obsDim {
            v += wVal[k] * obs[k]
        }
        return v
    }

    /// Descente de gradient sur \(L = -A \log \pi(a|s)\) (politique softmax masquée).
    public mutating func policyGradientStep(
        obs: [Float],
        mask: [Bool],
        action: Int,
        advantage: Float,
        lr: Float
    ) {
        let lg = logits(obs: obs, mask: mask)
        let probs = softmaxMaskedFromLogits(lg)
        let useParallel = SwiftComputeRuntime.threads > 1 && nAct >= 32
        if useParallel {
            wPi.withUnsafeMutableBufferPointer { wBuf in
                bPi.withUnsafeMutableBufferPointer { bBuf in
                    DispatchQueue.concurrentPerform(iterations: nAct) { i in
                        let gradLogit = probs[i] - (i == action ? 1.0 : 0.0)
                        let g = advantage * gradLogit
                        let row = i * obsDim
                        for k in 0..<obsDim {
                            wBuf[row + k] -= lr * g * obs[k]
                        }
                        bBuf[i] -= lr * g
                    }
                }
            }
        } else {
            for i in 0..<nAct {
                let gradLogit = probs[i] - (i == action ? 1.0 : 0.0)
                let g = advantage * gradLogit
                let row = i * obsDim
                for k in 0..<obsDim {
                    wPi[row + k] -= lr * g * obs[k]
                }
                bPi[i] -= lr * g
            }
        }
    }

    public mutating func valueStep(obs: [Float], targetReturn: Float, lr: Float) {
        let v = value(obs: obs)
        let err = v - targetReturn
        let g = 2.0 * err
        for k in 0..<obsDim {
            wVal[k] -= lr * g * obs[k]
        }
        bVal -= lr * g
    }
}

public func softmaxMaskedFromLogits(_ logits: [Float]) -> [Float] {
    var m = logits[0]
    for x in logits.dropFirst() {
        m = max(m, x)
    }
    let ex = logits.map { exp($0 - m) }
    let s = ex.reduce(0, +)
    return ex.map { $0 / s }
}

public func sampleCategorical(_ probs: [Float], rng: inout SplitMix64) -> Int {
    let u = Float(rng.nextDouble(in: 0...1))
    var c: Float = 0
    for i in 0..<probs.count {
        c += probs[i]
        if u <= c {
            return i
        }
    }
    return probs.count - 1
}
