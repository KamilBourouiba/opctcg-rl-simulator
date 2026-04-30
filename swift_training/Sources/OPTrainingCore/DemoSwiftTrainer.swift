import Foundation

/// Démo d’optimisation **sans MLX** : SGD sur une régression linéaire \( \hat y = w\cdot x + b \)
/// avec données synthétiques (reproductibles via ``TrainingConfig.seed``).
///
/// Sert de squelette « rules JSON → boucle d’apprentissage » jusqu’à branchement MLX ou simulateur OPTCG.
public enum SwiftCPUDemoTrainer {

    /// Retourne la MSE moyenne sur tout le run.
    public static func run(rules: TrainingRules, logEvery: Int = 400) -> Float {
        let cfg = rules.training
        let od = max(1, cfg.obsDim)
        var rng = SplitMix64(seed: cfg.seed)

        var w = [Float](repeating: 0, count: od)
        for i in 0..<od {
            w[i] = Float(rng.nextDouble(in: -0.05...0.05))
        }
        var b = Float(0)

        let lr = cfg.lr
        let steps = min(max(100, cfg.totalSteps), 100_000)

        var sumSq: Float = 0

        for t in 0..<steps {
            var x = [Float](repeating: 0, count: od)
            for i in 0..<od {
                x[i] = Float(rng.nextGaussian())
            }

            let target =
                x.reduce(0, +) * (1.0 / Float(od)) * 0.12
                    + Float(rng.nextDouble(in: -0.02...0.02))

            var pred = b
            for i in 0..<od {
                pred += w[i] * x[i]
            }

            let err = pred - target
            sumSq += err * err

            for i in 0..<od {
                w[i] -= lr * err * x[i]
            }
            b -= lr * err

            if logEvery > 0, t % logEvery == 0 || t == steps - 1 {
                let mseAvg = sumSq / Float(t + 1)
                fputs(
                    "[optrain] step \(t + 1)/\(steps)  mse_avg=\(String(format: "%.6f", mseAvg))\n",
                    stderr,
                )
            }
        }

        return sumSq / Float(max(1, steps))
    }
}
