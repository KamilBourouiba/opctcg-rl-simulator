import Foundation
import OPCTCGPolicy

/// Si le dossier contient ``coreml_manifest.json`` + ``PolicyPi.mlpackage`` (export
/// ``scripts/export_policy_coreml.py``), l'inférence passe par **Core ML** (ANE).
/// Sinon bundle ``manifest.json`` + ``weights.bin`` (Swift float).
///
/// Exemple :
///   swift run op-policy-infer ./swift_export "0.1,0.2,..."
/// Masque : même longueur que ``n_act``, caractères ``1`` = légal, ``0`` = illégal :
///   swift run op-policy-infer ./swift_export "..." --mask "111110..."
///
/// Sortie : une ligne JSON avec ``logits``, ``value``, ``argmax_masked``.
enum CLIError: Error {
    case usage
}

func parseFloats(_ s: String) throws -> [Float] {
    let parts = s.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
    return try parts.map {
        guard let v = Float($0) else { throw CLIError.usage }
        return v
    }
}

func parseMask(_ s: String, length: Int) throws -> [Bool] {
    let t = s.trimmingCharacters(in: .whitespaces)
    guard t.count == length else { throw CLIError.usage }
    return t.map { ch in ch == "1" }
}

@main
struct OpPolicyInferApp {
    static func main() {
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count >= 2 else {
                fputs("usage: op-policy-infer <bundle_dir> <obs_csv> [--mask 10101...]\n", stderr)
                exit(1)
            }
            let bundleURL = URL(fileURLWithPath: args[0], isDirectory: true)
            let obs = try parseFloats(args[1])
            let fm = FileManager.default
            let coremlManifestURL = bundleURL.appendingPathComponent("coreml_manifest.json")
            let piPackURL = bundleURL.appendingPathComponent("PolicyPi.mlpackage")

            let maskArgIdx = args.firstIndex(of: "--mask")
            var maskStr: String?
            if let idx = maskArgIdx, idx + 1 < args.count {
                maskStr = args[idx + 1]
            }

            let out: [String: Any]

            if fm.fileExists(atPath: coremlManifestURL.path), fm.fileExists(atPath: piPackURL.path) {
                let engine = try SwiftPolicyCoreMLEngine(bundleDirectory: bundleURL)
                let na = engine.manifest.n_act
                let mask: [Bool]
                if let ms = maskStr {
                    mask = try parseMask(ms, length: na)
                } else {
                    mask = Array(repeating: true, count: na)
                }
                let (logits, value) = try engine.forward(observation: obs, legalMask: mask)
                let argmax = try engine.greedyAction(observation: obs, legalMask: mask)
                out = [
                    "backend": "coreml",
                    "value": value,
                    "argmax_masked": argmax,
                    "logits": logits,
                ]
            } else {
                let policy = try SwiftPolicyBundle(bundleDirectory: bundleURL)
                let na = policy.manifest.n_act
                let mask: [Bool]
                if let ms = maskStr {
                    mask = try parseMask(ms, length: na)
                } else {
                    mask = Array(repeating: true, count: na)
                }
                let (logits, value) = try policy.forward(observation: obs, legalMask: mask)
                let argmax = try policy.greedyAction(observation: obs, legalMask: mask)
                out = [
                    "backend": "swift_float",
                    "value": value,
                    "argmax_masked": argmax,
                    "logits": logits,
                ]
            }
            let data = try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted])
            FileHandle.standardOutput.write(data)
            FileHandle.standardOutput.write(Data([10]))
        } catch {
            fputs("error: \(error)\n", stderr)
            exit(1)
        }
    }
}
