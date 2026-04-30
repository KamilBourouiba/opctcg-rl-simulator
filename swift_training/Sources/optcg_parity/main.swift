import Foundation
import OPTrainingCore
import OPTCGBridge
import OPTCGSimulator

@main
struct OptcgParityMain {
    static func main() {
        var repo = ".."
        var python = "/usr/bin/python3"
        var script = "scripts/swift_env_server.py"
        var cardsCsv = "data/cards_tcgcsv.csv"
        var deck0 = "decks/tournament_op15/p0001_Luigi_Amato.txt"
        var deck1 = "decks/tournament_op15/p0002_Progress2126.txt"
        var episodes = 5
        var maxSteps = 128
        var obsDim = GameConstants.defaultObsDim
        var seedBase = 1

        let argv = Array(CommandLine.arguments.dropFirst())
        var i = 0
        while i < argv.count {
            switch argv[i] {
            case "--repo": repo = argv[i + 1]; i += 2
            case "--python": python = argv[i + 1]; i += 2
            case "--script": script = argv[i + 1]; i += 2
            case "--cards-csv": cardsCsv = argv[i + 1]; i += 2
            case "--deck0": deck0 = argv[i + 1]; i += 2
            case "--deck1": deck1 = argv[i + 1]; i += 2
            case "--episodes": episodes = Int(argv[i + 1]) ?? episodes; i += 2
            case "--max-steps": maxSteps = Int(argv[i + 1]) ?? maxSteps; i += 2
            case "--obs-dim": obsDim = Int(argv[i + 1]) ?? obsDim; i += 2
            case "--seed-base": seedBase = Int(argv[i + 1]) ?? seedBase; i += 2
            default:
                i += 1
            }
        }

        let root = URL(fileURLWithPath: repo).standardizedFileURL
        let py = URL(fileURLWithPath: python)
        let srv = root.appendingPathComponent(script)
        var rng = SplitMix64(seed: UInt64(seedBase))
        var diverged = 0

        for ep in 0..<episodes {
            let seed = seedBase + ep
            do {
                let pyEnv = try PythonEnvSession(pythonExecutable: py, serverScript: srv, workingDirectory: root)
                    _ = try pyEnv.create(
                        repoRoot: root,
                        deck0: deck0,
                        deck1: deck1,
                        obsDim: obsDim,
                        seed: seed,
                        parityNoShuffle: true
                    )
                let swEnv = try NativeTextSimSession(
                    repoRoot: root,
                    deck0: deck0,
                    deck1: deck1,
                    cardsCsvRelativePath: cardsCsv,
                    columnMap: [:],
                    obsDim: obsDim,
                        seed: seed,
                        shuffleDecks: false,
                        selfPlay: false,
                        enableTextEffects: false
                )

                var pyState = try pyEnv.reset(seed: seed)
                var swState = try swEnv.reset(seed: seed)
                var localDiverged = false
                var reason = ""

                for stepIdx in 0..<maxSteps {
                    let sameMask = (pyState.mask.count == swState.mask.count) && zip(pyState.mask, swState.mask).allSatisfy { $0 == $1 }
                    if !sameMask {
                        localDiverged = true
                        let diff = firstMaskDiffIndices(py: pyState.mask, sw: swState.mask, limit: 12)
                        let pyCount = pyState.mask.filter { $0 }.count
                        let swCount = swState.mask.filter { $0 }.count
                        var detail = "mask_mismatch@\(stepIdx):\(diff) pyCount=\(pyCount) swCount=\(swCount) py45=\(boolAt(pyState.mask, 45)) sw45=\(boolAt(swState.mask, 45))"
                        if let dbg = try? pyEnv.debugState(),
                           let hand = dbg["p0_hand"] as? [String],
                           let costs = dbg["p0_hand_costs"] as? [Int],
                           let types = dbg["p0_hand_types"] as? [String],
                           let phase = dbg["phase"] as? String {
                            let pairs = zip(hand, zip(costs, types)).prefix(7).map { h, ct in "\(h):\(ct.0):\(ct.1)" }
                            detail += " py_phase=\(phase) py_hand=\(pairs.joined(separator: "|")) py_don=\(dbg["p0_don_active"] ?? -1)"
                        }
                        let swSnap = swEnv.replayState()
                        let swHand = swSnap.p0.hand.prefix(7).joined(separator: "|")
                        detail += " sw_phase=\(swSnap.phase) sw_hand=\(swHand) sw_don=\(swSnap.p0.donActive)"
                        reason = detail
                        break
                    }
                    let legal = sharedLegalActions(pyMask: pyState.mask, swMask: swState.mask)
                    if legal.isEmpty {
                        localDiverged = true
                        reason = "no_shared_legal@\(stepIdx)"
                        break
                    }
                    let a = legal[Int(rng.nextUInt64() % UInt64(legal.count))]
                    let pyStep = try pyEnv.step(action: a)
                    let swStep = try swEnv.step(action: a)
                    if pyStep.done != swStep.done {
                        localDiverged = true
                        reason = "done_mismatch@\(stepIdx)"
                        break
                    }
                    pyState = (pyStep.obs, pyStep.mask)
                    swState = (swStep.obs, swStep.mask)
                    if pyStep.done { break }
                }
                try? pyEnv.close()
                try? swEnv.close()
                if localDiverged {
                    diverged += 1
                    fputs("[parity] episode \(ep + 1): DIVERGED \(reason)\n", stderr)
                } else {
                    fputs("[parity] episode \(ep + 1): OK\n", stderr)
                }
            } catch {
                diverged += 1
                fputs("[parity] episode \(ep + 1): ERROR \(error)\n", stderr)
            }
        }
        print("{\"ok\":true,\"episodes\":\(episodes),\"diverged\":\(diverged)}")
    }

    private static func sharedLegalActions(pyMask: [Bool], swMask: [Bool]) -> [Int] {
        let n = min(pyMask.count, swMask.count)
        var out: [Int] = []
        out.reserveCapacity(32)
        for i in 0..<n where pyMask[i] && swMask[i] {
            out.append(i)
        }
        return out
    }

    private static func firstMaskDiffIndices(py: [Bool], sw: [Bool], limit: Int) -> String {
        let n = min(py.count, sw.count)
        var out: [String] = []
        for i in 0..<n where py[i] != sw[i] {
            out.append(String(i))
            if out.count >= limit { break }
        }
        if py.count != sw.count {
            out.append("len:\(py.count)-\(sw.count)")
        }
        return out.joined(separator: ",")
    }

    private static func boolAt(_ a: [Bool], _ idx: Int) -> Int {
        guard idx >= 0, idx < a.count else { return -1 }
        return a[idx] ? 1 : 0
    }
}
