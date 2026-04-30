import CoreML
import Foundation

/// Manifest écrit par ``scripts/export_policy_coreml.py`` (Python).
public struct CoreMLPolicyManifest: Codable {
    public let schema_version: Int
    public let obs_dim: Int
    public let n_act: Int
    public let input_name: String
    public let neg_inf_mask: Float
    public let packages: CoreMLPackageNames
}

public struct CoreMLPackageNames: Codable {
    public let pi: String
    public let val: String
}

/// Inférence via **Core ML** (ANE / GPU / CPU) — en général bien plus rapide que la boucle Swift float
/// pour les batchs ou les inférences répétées.
public final class SwiftPolicyCoreMLEngine {
    private let piModel: MLModel
    private let valModel: MLModel
    public let manifest: CoreMLPolicyManifest

    public init(bundleDirectory: URL) throws {
        let manURL = bundleDirectory.appendingPathComponent("coreml_manifest.json")
        let dec = JSONDecoder()
        let data = try Data(contentsOf: manURL)
        self.manifest = try dec.decode(CoreMLPolicyManifest.self, from: data)

        let cfg = MLModelConfiguration()
        cfg.computeUnits = .all

        let piURL = bundleDirectory.appendingPathComponent(manifest.packages.pi)
        let valURL = bundleDirectory.appendingPathComponent(manifest.packages.val)

        let compiledPi = try MLModel.compileModel(at: piURL)
        let compiledVal = try MLModel.compileModel(at: valURL)
        self.piModel = try MLModel(contentsOf: compiledPi, configuration: cfg)
        self.valModel = try MLModel(contentsOf: compiledVal, configuration: cfg)
    }

    private func inputProvider(observation: [Float]) throws -> MLFeatureProvider {
        precondition(observation.count == manifest.obs_dim)
        let shape = [1, manifest.obs_dim] as [NSNumber]
        let arr = try MLMultiArray(shape: shape, dataType: .float32)
        for i in 0..<observation.count {
            arr[[0, NSNumber(value: i)] as [NSNumber]] = NSNumber(value: observation[i])
        }
        return try MLDictionaryFeatureProvider(dictionary: [
            manifest.input_name: MLFeatureValue(multiArray: arr),
        ])
    }

    private static func firstMultiArray(from prediction: MLFeatureProvider) throws -> MLMultiArray {
        for name in prediction.featureNames {
            if let v = prediction.featureValue(for: name)?.multiArrayValue {
                return v
            }
        }
        throw SwiftPolicyError.missingTensor("coreml_output")
    }

    /// Logits complets + valeur ; applique le masque illégal comme PyTorch / MLX.
    public func forward(observation: [Float], legalMask: [Bool]) throws -> (logits: [Float], value: Float) {
        guard legalMask.count == manifest.n_act else {
            throw SwiftPolicyError.badMask(got: legalMask.count, expected: manifest.n_act)
        }

        let inPi = try inputProvider(observation: observation)
        let inVal = try inputProvider(observation: observation)

        let predPi = try piModel.prediction(from: inPi)
        let predVal = try valModel.prediction(from: inVal)

        let logitsArr = try Self.firstMultiArray(from: predPi)
        let valArr = try Self.firstMultiArray(from: predVal)

        let na = manifest.n_act
        var logits = [Float](repeating: 0, count: na)
        for i in 0..<na {
            logits[i] = Float(truncating: logitsArr[[0, NSNumber(value: i)] as [NSNumber]])
            if !legalMask[i] {
                logits[i] = manifest.neg_inf_mask
            }
        }

        let value = Float(truncating: valArr[0])
        return (logits, value)
    }

    public func greedyAction(observation: [Float], legalMask: [Bool]) throws -> Int {
        let (logits, _) = try forward(observation: observation, legalMask: legalMask)
        var best = -1
        var bestv = manifest.neg_inf_mask
        for i in 0..<logits.count where legalMask[i] {
            if logits[i] > bestv {
                bestv = logits[i]
                best = i
            }
        }
        guard best >= 0 else { throw SwiftPolicyError.noLegalAction }
        return best
    }
}
