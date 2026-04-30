import Foundation
import OPTrainingCore
import OPTCGBridge
import OPTCGRL
import OPTCGSimulator

private struct SavedLinearModel: Codable {
    var format: String
    var obsDim: Int
    var nAct: Int
    var wPi: [Float]
    var bPi: [Float]
    var wVal: [Float]
    var bVal: Float
}

private struct ReplayFrame: Codable {
    var frame: Int
    var kind: String
    var phase: String
    var action: Int?
    var actionDesc: String?
    var reward: Float?
    var done: Bool
    var winner: Int?
    var p0: NativeSimulator.ReplayPlayer
    var p1: NativeSimulator.ReplayPlayer
    var simMeta: [String: String]

    enum CodingKeys: String, CodingKey {
        case frame
        case kind
        case phase
        case action
        case actionDesc = "action_desc"
        case reward
        case done
        case winner
        case p0
        case p1
        case simMeta = "sim_meta"
    }
}

private func saveModel(_ net: LinearActorCritic, to path: URL) throws {
    let payload = SavedLinearModel(
        format: "optcg_swift_linear_actor_critic_v1",
        obsDim: net.obsDim,
        nAct: net.nAct,
        wPi: net.wPi,
        bPi: net.bPi,
        wVal: net.wVal,
        bVal: net.bVal,
    )
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try enc.encode(payload)
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: path)
}

private func writeReplayJsonl(_ frames: [ReplayFrame], to path: URL) throws {
    let enc = JSONEncoder()
    var out = Data()
    for fr in frames {
        out.append(try enc.encode(fr))
        out.append(Data([10]))
    }
    try FileManager.default.createDirectory(at: path.deletingLastPathComponent(), withIntermediateDirectories: true)
    try out.write(to: path)
}

private func uniqueOutputURL(_ desired: URL) -> URL {
    let fm = FileManager.default
    guard fm.fileExists(atPath: desired.path) else { return desired }
    let dir = desired.deletingLastPathComponent()
    let ext = desired.pathExtension
    let base = desired.deletingPathExtension().lastPathComponent
    let fmt = DateFormatter()
    fmt.locale = Locale(identifier: "en_US_POSIX")
    fmt.timeZone = TimeZone.current
    fmt.dateFormat = "yyyyMMdd_HHmmss"
    let ts = fmt.string(from: Date())
    let primary = ext.isEmpty ? "\(base)_\(ts)" : "\(base)_\(ts).\(ext)"
    let primaryURL = dir.appendingPathComponent(primary)
    guard fm.fileExists(atPath: primaryURL.path) else { return primaryURL }
    // Très improbable, mais si deux sorties tombent la même seconde, on disambiguïse.
    for i in 1...999 {
        let name = ext.isEmpty ? "\(base)_\(ts)_\(i)" : "\(base)_\(ts)_\(i).\(ext)"
        let cand = dir.appendingPathComponent(name)
        if !fm.fileExists(atPath: cand.path) { return cand }
    }
    return primaryURL
}

private func renderReplayMp4(
    repoRoot: URL,
    jsonlPath: URL,
    mp4Path: URL,
    python: String,
    cardsCsvRelativePath: String,
) throws {
    let pySnippet = """
import csv
import json
from pathlib import Path
from opctcg_text_sim.replay_render_images import jsonl_to_mp4_with_images
root = Path(r'\(repoRoot.path)')
jsonl = Path(r'\(jsonlPath.path)')
mp4 = Path(r'\(mp4Path.path)')
manifest = root / "data" / "cards_manifest.json"
cards_csv = root / Path(r'\(cardsCsvRelativePath)')
if (not manifest.exists()) and cards_csv.exists():
    rows = {}
    with cards_csv.open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            cid = (r.get("card_id") or r.get("id") or "").strip().upper()
            if not cid:
                continue
            rows[cid] = {
                "name": (r.get("name") or "").strip(),
                "image_url": (r.get("image_url") or r.get("imageUrl") or "").strip(),
                "card_text": (r.get("card_text") or "").strip(),
            }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
cache_dir = root / "data" / "card_image_cache"
jsonl_to_mp4_with_images(jsonl, mp4, manifest, cache_dir)
print(str(mp4))
"""
    let proc = Process()
    proc.executableURL = URL(fileURLWithPath: python)
    proc.arguments = ["-c", pySnippet]
    proc.currentDirectoryURL = repoRoot
    let out = Pipe()
    proc.standardOutput = out
    proc.standardError = out
    try proc.run()
    proc.waitUntilExit()
    if proc.terminationStatus != 0 {
        let msg = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "render error"
        throw NSError(domain: "optcg_train", code: Int(proc.terminationStatus), userInfo: [NSLocalizedDescriptionKey: msg])
    }
}

private func greedyAction(net: LinearActorCritic, obs: [Float], mask: [Bool]) -> Int {
    let lg = net.logits(obs: obs, mask: mask)
    var bestIdx = 0
    var best = lg[0]
    for i in 1..<lg.count where lg[i] > best {
        best = lg[i]
        bestIdx = i
    }
    return bestIdx
}

private func collectNativeReplay(
    session: NativeTextSimSession,
    net: LinearActorCritic,
    maxSteps: Int,
    seed: Int,
) throws -> [ReplayFrame] {
    var frames: [ReplayFrame] = []
    var obsMask = try session.reset(seed: seed)
    let initState = session.replayState()
    frames.append(
        ReplayFrame(
            frame: 0,
            kind: "reset",
            phase: initState.phase,
            action: nil,
            actionDesc: nil,
            reward: nil,
            done: initState.done,
            winner: initState.winner,
            p0: initState.p0,
            p1: initState.p1,
            simMeta: ["rules_version": "native_v1", "max_don": String(initState.p0.officialDonDeck)],
        )
    )
    for stepIdx in 0..<maxSteps {
        let a = greedyAction(net: net, obs: obsMask.obs, mask: obsMask.mask)
        let st = try session.step(action: a)
        let snap = session.replayState()
        frames.append(
            ReplayFrame(
                frame: stepIdx + 1,
                kind: "step",
                phase: snap.phase,
                action: a,
                actionDesc: session.actionDescription(a),
                reward: st.reward,
                done: st.done,
                winner: snap.winner,
                p0: snap.p0,
                p1: snap.p1,
                simMeta: ["rules_version": "native_v1", "max_don": String(snap.p0.officialDonDeck)],
            )
        )
        if st.done { break }
        obsMask = (st.obs, st.mask)
    }
    return frames
}

private func listDeckFiles(in root: URL, relativeDir: String, pattern: String) -> [String] {
    let dir = root.appendingPathComponent(relativeDir)
    guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
        return []
    }
    let regexPattern = "^" + NSRegularExpression.escapedPattern(for: pattern)
        .replacingOccurrences(of: "\\*", with: ".*")
        .replacingOccurrences(of: "\\?", with: ".") + "$"
    let re = try? NSRegularExpression(pattern: regexPattern, options: [.caseInsensitive])
    return files
        .filter { $0.pathExtension.lowercased() == "txt" }
        .map { $0.lastPathComponent }
        .filter { name in
            guard let re else { return true }
            let ns = name as NSString
            return re.firstMatch(in: name, range: NSRange(location: 0, length: ns.length)) != nil
        }
        .sorted()
        .map { "\(relativeDir.trimmingCharacters(in: CharacterSet(charactersIn: "/")))/\($0)" }
}

/// Une passe REINFORCE + baseline sur une partie utilisant le serveur Python (simulateur complet).
func trainEpisodeRPC(
    env: TextSimEnvironment,
    net: inout LinearActorCritic,
    gamma: Float,
    lrPolicy: Float,
    lrValue: Float,
    maxSteps: Int,
    rng: inout SplitMix64,
    seed: Int
) throws -> Float {
    var obsMask = try env.reset(seed: seed)
    var transitions: [(obs: [Float], mask: [Bool], action: Int, reward: Float)] = []
    var totalReward: Float = 0

    for _ in 0..<maxSteps {
        let lg = net.logits(obs: obsMask.obs, mask: obsMask.mask)
        let probs = softmaxMaskedFromLogits(lg)
        let a = sampleCategorical(probs, rng: &rng)
        let step = try env.step(action: a)
        totalReward += step.reward
        transitions.append((obsMask.obs, obsMask.mask, a, step.reward))
        if step.done {
            break
        }
        obsMask = (step.obs, step.mask)
    }

    let n = transitions.count
    guard n > 0 else { return 0 }

    var returns = [Float](repeating: 0, count: n)
    var g: Float = 0
    for t in (0..<n).reversed() {
        g = transitions[t].reward + gamma * g
        returns[t] = g
    }

    for t in 0..<n {
        let tr = transitions[t]
        let adv = returns[t] - net.value(obs: tr.obs)
        net.policyGradientStep(
            obs: tr.obs,
            mask: tr.mask,
            action: tr.action,
            advantage: adv,
            lr: lrPolicy,
        )
        net.valueStep(obs: tr.obs, targetReturn: returns[t], lr: lrValue)
    }

    return totalReward
}

enum CLIExit: Int32 {
    case ok = 0
    case usage = 1
}

@main
struct OptcgTrainMain {
    static func main() {
        var repoRoot: String?
        var python = "/usr/bin/python3"
        var scriptRel = "scripts/swift_env_server.py"
        var cardsCsvRel = "data/cards_tcgcsv.csv"
        var backend = "python"
        var deck0 = "decks/NAMIBY.txt"
        var deck1: String?
        var gauntletDir = "decks/tournament_op15"
        var gauntletGlob = "p*.txt"
        var tournamentOnly = true
        var episodes = 30
        var maxSteps = 2048
        var obsDim = GameConstants.defaultObsDim
        var seedBase = 1
        var gamma: Float = 0.995
        var lrPolicy: Float = 2e-4
        var lrValue: Float = 5e-4
        var threadsArg = "auto"
        var saveModelRel: String?
        var replayJsonlRel: String?
        var videoMp4Rel: String?

        let argv = Array(CommandLine.arguments.dropFirst())
        var i = argv.startIndex
        while i < argv.endIndex {
            switch argv[i] {
            case "--repo", "-r":
                guard i + 1 < argv.endIndex else { fputs("--repo requiert un chemin\n", stderr); exit(CLIExit.usage.rawValue) }
                repoRoot = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--python":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                python = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--script":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                scriptRel = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--cards-csv":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                cardsCsvRel = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--backend":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                backend = argv[i + 1].lowercased()
                i = argv.index(i, offsetBy: 2)
            case "--deck0":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                deck0 = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--deck1":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                deck1 = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--gauntlet-dir":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                gauntletDir = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--gauntlet-glob":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                gauntletGlob = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--tournament-only":
                tournamentOnly = true
                i = argv.index(after: i)
            case "--no-tournament-only":
                tournamentOnly = false
                i = argv.index(after: i)
            case "--episodes":
                guard i + 1 < argv.endIndex, let v = Int(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                episodes = v
                i = argv.index(i, offsetBy: 2)
            case "--max-steps":
                guard i + 1 < argv.endIndex, let v = Int(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                maxSteps = v
                i = argv.index(i, offsetBy: 2)
            case "--obs-dim":
                guard i + 1 < argv.endIndex, let v = Int(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                obsDim = v
                i = argv.index(i, offsetBy: 2)
            case "--seed-base":
                guard i + 1 < argv.endIndex, let v = Int(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                seedBase = v
                i = argv.index(i, offsetBy: 2)
            case "--gamma":
                guard i + 1 < argv.endIndex, let v = Float(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                gamma = v
                i = argv.index(i, offsetBy: 2)
            case "--lr-policy":
                guard i + 1 < argv.endIndex, let v = Float(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                lrPolicy = v
                i = argv.index(i, offsetBy: 2)
            case "--lr-value":
                guard i + 1 < argv.endIndex, let v = Float(argv[i + 1]) else { exit(CLIExit.usage.rawValue) }
                lrValue = v
                i = argv.index(i, offsetBy: 2)
            case "--threads":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                threadsArg = argv[i + 1].lowercased()
                i = argv.index(i, offsetBy: 2)
            case "--save-model":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                saveModelRel = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--replay-jsonl":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                replayJsonlRel = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--video-mp4":
                guard i + 1 < argv.endIndex else { exit(CLIExit.usage.rawValue) }
                videoMp4Rel = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
            case "--help", "-h":
                print(
                    """
                    optcg_train — REINFORCE Swift + simulateur Python/Natif

                      swift run optcg_train --repo /chemin/opctcg_text_sim

                    Options :
                      --repo        Racine du dépôt (config.yaml, decks/, …)
                      --python      Interpréteur Python (défaut /usr/bin/python3)
                      --script      Relatif au repo : scripts/swift_env_server.py
                      --cards-csv   Relatif au repo : data/cards_tcgcsv.csv
                      --backend     python|native (défaut python)
                      --deck0/--deck1  Chemins decks relatifs au repo
                      --gauntlet-dir Dossier decks tournoi (défaut decks/tournament_op15)
                      --gauntlet-glob Filtre deck tournoi (défaut p*.txt)
                      --tournament-only / --no-tournament-only
                      --episodes    Nombre de parties (défaut 30)
                      --max-steps   Plafond pas par partie (défaut 2048)
                      --obs-dim     Doit matcher training.obs_dim (défaut 96)
                      --threads     auto|N (défaut auto, utilise tous les cœurs)
                      --save-model  Sortie JSON du modèle Swift (relatif repo)
                      --replay-jsonl  Replay greedy (native uniquement)
                      --video-mp4   MP4 rendu depuis replay (native uniquement)
                    """
                )
                exit(CLIExit.ok.rawValue)
            default:
                fputs("argument inconnu : \(argv[i])\n", stderr)
                exit(CLIExit.usage.rawValue)
            }
        }

        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let resolvedRepo: URL
        if let rs = repoRoot {
            resolvedRepo = URL(fileURLWithPath: rs).standardizedFileURL
        } else {
            resolvedRepo = cwd
        }

        var rng = SplitMix64(seed: UInt64(seedBase))
        var nAct = GameConstants.actionSpaceSize
        var effectiveObsDim = obsDim

        if threadsArg == "auto" {
            SwiftComputeRuntime.autoConfigure()
        } else if let n = Int(threadsArg), n > 0 {
            SwiftComputeRuntime.setThreads(n)
        } else {
            fputs("valeur --threads invalide : \(threadsArg) (attendu: auto|N)\n", stderr)
            exit(1)
        }
        fputs("[optcg_train] compute threads=\(SwiftComputeRuntime.threads)\n", stderr)

        let tournamentDecks: [String] = tournamentOnly ? listDeckFiles(
            in: resolvedRepo,
            relativeDir: gauntletDir,
            pattern: gauntletGlob,
        ) : []
        if tournamentOnly {
            guard tournamentDecks.count >= 2 else {
                fputs("aucun pool tournoi exploitable dans \(gauntletDir) (glob=\(gauntletGlob))\n", stderr)
                exit(1)
            }
            fputs("[optcg_train] tournament-only ON  decks=\(tournamentDecks.count)  dir=\(gauntletDir)  glob=\(gauntletGlob)\n", stderr)
        }

        do {
            var net = LinearActorCritic(
                obsDim: effectiveObsDim,
                nAct: nAct,
                rng: &rng,
            )

            var rewards = [Float]()
            var replayDeckPair: (String, String)?
            for ep in 0..<episodes {
                let epDeck0: String
                let epDeck1: String?
                if tournamentOnly {
                    let i0 = Int(rng.nextUInt64() % UInt64(tournamentDecks.count))
                    var i1 = Int(rng.nextUInt64() % UInt64(tournamentDecks.count))
                    if i1 == i0 { i1 = (i1 + 1) % tournamentDecks.count }
                    epDeck0 = tournamentDecks[i0]
                    epDeck1 = tournamentDecks[i1]
                } else {
                    epDeck0 = deck0
                    epDeck1 = deck1
                }

                let env: TextSimEnvironment
                do {
                    if backend == "python" {
                        let py = URL(fileURLWithPath: python)
                        let srv = resolvedRepo.appendingPathComponent(scriptRel)
                        guard FileManager.default.fileExists(atPath: srv.path) else {
                            fputs("script introuvable : \(srv.path)\n", stderr)
                            exit(1)
                        }
                        let session = try PythonEnvSession(
                            pythonExecutable: py,
                            serverScript: srv,
                            workingDirectory: resolvedRepo,
                        )
                        let meta = try session.create(
                            repoRoot: resolvedRepo,
                            deck0: epDeck0,
                            deck1: epDeck1,
                            obsDim: obsDim,
                            seed: seedBase + ep,
                        )
                        nAct = meta.actionSpace
                        effectiveObsDim = meta.obsDim
                        env = session
                    } else if backend == "native" {
                        env = try NativeTextSimSession(
                            repoRoot: resolvedRepo,
                            deck0: epDeck0,
                            deck1: epDeck1,
                            cardsCsvRelativePath: cardsCsvRel,
                            columnMap: [:],
                            obsDim: obsDim,
                            seed: seedBase + ep,
                        )
                    } else {
                        fputs("backend inconnu : \(backend) (attendu: python|native)\n", stderr)
                        exit(1)
                    }
                } catch {
                    fputs("init backend (\(backend)) : \(error)\n", stderr)
                    exit(1)
                }

                let rew = try trainEpisodeRPC(
                    env: env,
                    net: &net,
                    gamma: gamma,
                    lrPolicy: lrPolicy,
                    lrValue: lrValue,
                    maxSteps: maxSteps,
                    rng: &rng,
                    seed: seedBase + ep,
                )
                try? env.close()
                rewards.append(rew)
                if replayDeckPair == nil {
                    replayDeckPair = (epDeck0, epDeck1 ?? epDeck0)
                }
                if ep % max(1, episodes / 10) == 0 || ep == episodes - 1 {
                    let tail = rewards.suffix(10).reduce(0, +) / Float(min(10, rewards.count))
                    fputs(
                        "[optcg_train] ep \(ep + 1)/\(episodes)  R=\(String(format: "%.3f", rew))  avg10=\(String(format: "%.3f", tail))  deck0=\(epDeck0)  deck1=\(epDeck1 ?? epDeck0)\n",
                        stderr,
                    )
                }
            }

            if let rel = saveModelRel {
                let path = resolvedRepo.appendingPathComponent(rel)
                try saveModel(net, to: path)
                fputs("[optcg_train] modèle sauvegardé : \(path.path)\n", stderr)
            }

            if replayJsonlRel != nil || videoMp4Rel != nil {
                guard backend == "native" else {
                    fputs("[optcg_train] replay/video disponibles seulement avec --backend native\n", stderr)
                    exit(1)
                }
                let pair = replayDeckPair ?? (deck0, deck1 ?? deck0)
                let native = try NativeTextSimSession(
                    repoRoot: resolvedRepo,
                    deck0: pair.0,
                    deck1: pair.1,
                    cardsCsvRelativePath: cardsCsvRel,
                    columnMap: [:],
                    obsDim: obsDim,
                    seed: seedBase + episodes + 1,
                )
                let replayRel = replayJsonlRel ?? "runs/native_replay.jsonl"
                let replayPath = uniqueOutputURL(resolvedRepo.appendingPathComponent(replayRel))
                let frames = try collectNativeReplay(
                    session: native,
                    net: net,
                    maxSteps: maxSteps,
                    seed: seedBase + episodes + 1,
                )
                try writeReplayJsonl(frames, to: replayPath)
                try? native.close()
                fputs("[optcg_train] replay jsonl : \(replayPath.path)\n", stderr)

                if let mp4Rel = videoMp4Rel {
                    let mp4Path = uniqueOutputURL(resolvedRepo.appendingPathComponent(mp4Rel))
                    do {
                        try renderReplayMp4(
                            repoRoot: resolvedRepo,
                            jsonlPath: replayPath,
                            mp4Path: mp4Path,
                            python: python,
                            cardsCsvRelativePath: cardsCsvRel,
                        )
                        fputs("[optcg_train] vidéo mp4 : \(mp4Path.path)\n", stderr)
                    } catch {
                        fputs("[optcg_train] rendu vidéo échoué : \(error)\n", stderr)
                    }
                }
            }

            let meanR = rewards.reduce(0, +) / Float(max(1, rewards.count))
            print("{\"ok\":true,\"episodes\":\(episodes),\"mean_reward\":\(meanR)}")
            exit(CLIExit.ok.rawValue)
        } catch {
            fputs("entraînement : \(error)\n", stderr)
            exit(1)
        }
    }
}
