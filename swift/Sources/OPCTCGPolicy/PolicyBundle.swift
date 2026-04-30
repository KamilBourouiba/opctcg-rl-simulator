import Foundation

/// Charge ``manifest.json`` + ``weights.bin`` produits par ``scripts/export_policy_swift_bundle.py``.
public final class SwiftPolicyBundle {
    public let manifest: SwiftPolicyManifest
    private var tensors: [String: [Float]] = [:]

    public init(bundleDirectory: URL) throws {
        let manifestURL = bundleDirectory.appendingPathComponent("manifest.json")
        let data = try Data(contentsOf: manifestURL)
        let dec = JSONDecoder()
        self.manifest = try dec.decode(SwiftPolicyManifest.self, from: data)
        let weightsName = manifest.weights_file
        let weightsURL = bundleDirectory.appendingPathComponent(weightsName)
        let raw = try Data(contentsOf: weightsURL)
        guard manifest.dtype == "float32", manifest.endian == "little" else {
            throw SwiftPolicyError.unsupportedFormat
        }
        for seg in manifest.segments {
            let nFloats = seg.byte_length / 4
            let slice = raw.subdata(in: seg.byte_offset ..< (seg.byte_offset + seg.byte_length))
            guard slice.count == seg.byte_length else {
                throw SwiftPolicyError.segmentMismatch
            }
            let floats: [Float] = slice.withUnsafeBytes { raw in
                let p = raw.bindMemory(to: Float.self)
                return Array(UnsafeBufferPointer(start: p.baseAddress, count: nFloats))
            }
            tensors[seg.name] = floats
        }
    }

    private func tensor(_ name: String) throws -> [Float] {
        guard let t = tensors[name] else { throw SwiftPolicyError.missingTensor(name) }
        return t
    }

    /// Sortie alignée sur PyTorch / MLX : logits masqués + valeur scalaire.
    public func forward(observation: [Float], legalMask: [Bool]) throws -> (logits: [Float], value: Float) {
        guard observation.count == manifest.obs_dim else {
            throw SwiftPolicyError.badObsDim(got: observation.count, expected: manifest.obs_dim)
        }
        guard legalMask.count == manifest.n_act else {
            throw SwiftPolicyError.badMask(got: legalMask.count, expected: manifest.n_act)
        }

        let od = manifest.obs_dim
        let h = manifest.hidden_dim
        let na = manifest.n_act
        let nr = manifest.n_res_blocks

        let pw = try tensor("proj.linear.weight")
        let pb = try tensor("proj.linear.bias")
        let pg = try tensor("proj.norm.weight")
        let pbeta = try tensor("proj.norm.bias")

        var x = linearMatVec(x: observation, weight: pw, bias: pb, outDim: h, inDim: od)
        x = layerNormVec(x: x, gamma: pg, beta: pbeta)
        x = x.map { leakyRelu($0) }

        for i in 0..<nr {
            let l1w = try tensor("res\(i).lin1.weight")
            let l1b = try tensor("res\(i).lin1.bias")
            let n1w = try tensor("res\(i).ln1.weight")
            let n1b = try tensor("res\(i).ln1.bias")
            let l2w = try tensor("res\(i).lin2.weight")
            let l2b = try tensor("res\(i).lin2.bias")
            let n2w = try tensor("res\(i).ln2.weight")
            let n2b = try tensor("res\(i).ln2.bias")

            var y = linearMatVec(x: x, weight: l1w, bias: l1b, outDim: h, inDim: h)
            y = layerNormVec(x: y, gamma: n1w, beta: n1b)
            y = y.map { leakyRelu($0) }
            y = linearMatVec(x: y, weight: l2w, bias: l2b, outDim: h, inDim: h)
            y = layerNormVec(x: y, gamma: n2w, beta: n2b)
            x = zip(x, y).map { leakyRelu($0 + $1) }
        }

        let piW = try tensor("pi.weight")
        let piB = try tensor("pi.bias")
        let vW = try tensor("v.weight")
        let vB = try tensor("v.bias")

        var logits = linearMatVec(x: x, weight: piW, bias: piB, outDim: na, inDim: h)
        let vScalar = linearMatVec(x: x, weight: vW, bias: vB, outDim: 1, inDim: h)[0]

        let neg = manifest.neg_inf_mask
        for i in 0..<na where !legalMask[i] {
            logits[i] = neg
        }

        return (logits, vScalar)
    }

    /// \( \arg\max_a \text{logits}_a \) sur les actions légales uniquement.
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

public enum SwiftPolicyError: Error {
    case unsupportedFormat
    case missingTensor(String)
    case badObsDim(got: Int, expected: Int)
    case badMask(got: Int, expected: Int)
    case noLegalAction
    case segmentMismatch
}
