import Foundation
import OPTrainingCore

/// Lecture ligne à ligne depuis un ``FileHandle`` (stdout du sous-processus Python).
final class LineReader {
    private let fh: FileHandle
    private var buf = Data()

    init(fileHandle: FileHandle) {
        self.fh = fileHandle
    }

    func readLine() throws -> String {
        while true {
            if let range = buf.range(of: Data([10])) {
                let lineData = buf[..<range.lowerBound]
                buf.removeSubrange(..<range.upperBound)
                guard let s = String(data: Data(lineData), encoding: .utf8) else {
                    throw NSError(domain: "LineReader", code: 1, userInfo: [NSLocalizedDescriptionKey: "UTF-8"])
                }
                return s
            }
            let chunk = fh.availableData
            if chunk.isEmpty {
                if buf.isEmpty {
                    throw NSError(domain: "LineReader", code: 2, userInfo: [NSLocalizedDescriptionKey: "EOF"])
                }
                break
            }
            buf.append(chunk)
        }
        guard let s = String(data: buf, encoding: .utf8), !s.isEmpty else {
            throw NSError(domain: "LineReader", code: 3, userInfo: [NSLocalizedDescriptionKey: "empty"])
        }
        buf.removeAll()
        return s.trimmingCharacters(in: .newlines)
    }
}

/// Session « simulateur Python » — une ligne JSON par requête/réponse.
public final class PythonEnvSession {
    private let proc = Process()
    private let stdinFH: FileHandle
    private let lineReader: LineReader

    public init(pythonExecutable: URL, serverScript: URL, workingDirectory: URL? = nil) throws {
        proc.executableURL = pythonExecutable
        proc.arguments = [serverScript.path]
        if let wd = workingDirectory {
            proc.currentDirectoryURL = wd
        }

        let inPipe = Pipe()
        let outPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        if let devNull = FileHandle(forWritingAtPath: "/dev/null") {
            proc.standardError = devNull
        }

        try proc.run()

        guard let stdin = proc.standardInput as? Pipe else {
            throw NSError(domain: "PythonEnvSession", code: 1)
        }
        guard let stdout = proc.standardOutput as? Pipe else {
            throw NSError(domain: "PythonEnvSession", code: 2)
        }
        stdinFH = stdin.fileHandleForWriting
        lineReader = LineReader(fileHandle: stdout.fileHandleForReading)
    }

    deinit {
        stdinFH.closeFile()
        proc.terminate()
    }

    private func send(_ dict: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: dict)
        stdinFH.write(data)
        stdinFH.write(Data([10]))
    }

    private func recv() throws -> [String: Any] {
        let line = try lineReader.readLine()
        guard let data = line.data(using: .utf8),
              let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            throw NSError(domain: "PythonEnvSession", code: 3, userInfo: [NSLocalizedDescriptionKey: line])
        }
        return obj
    }

    public func create(
        repoRoot: URL,
        deck0: String,
        deck1: String?,
        obsDim: Int,
        seed: Int,
        parityNoShuffle: Bool = false
    ) throws -> (
        actionSpace: Int, obsDim: Int
    ) {
        var payload: [String: Any] = [
            "cmd": "create",
            "repo_root": repoRoot.path,
            "deck0": deck0,
            "obs_dim": obsDim,
            "seed": seed,
            "parity_no_shuffle": parityNoShuffle,
        ]
        if let d1 = deck1 {
            payload["deck1"] = d1
        }
        try send(payload)
        let r = try recv()
        guard let ok = r["ok"] as? Bool, ok else {
            throw NSError(domain: "PythonEnvSession", code: 10, userInfo: [NSLocalizedDescriptionKey: r["error"] ?? r])
        }
        let act = r["action_space"] as? Int ?? GameConstants.actionSpaceSize
        let od = r["obs_dim"] as? Int ?? obsDim
        return (act, od)
    }

    public func reset(seed: Int?) throws -> (obs: [Float], mask: [Bool]) {
        var payload: [String: Any] = ["cmd": "reset"]
        if let s = seed {
            payload["seed"] = s
        }
        try send(payload)
        let r = try recv()
        guard let ok = r["ok"] as? Bool, ok else {
            throw NSError(domain: "PythonEnvSession", code: 11, userInfo: [NSLocalizedDescriptionKey: r["error"] ?? r])
        }
        let obs = r["obs"] as? [Double]
        let mask = r["mask"] as? [Bool]
        guard let o = obs, let m = mask else {
            throw NSError(domain: "PythonEnvSession", code: 12)
        }
        return (o.map { Float($0) }, m)
    }

    public func step(action: Int) throws -> (obs: [Float], mask: [Bool], reward: Float, done: Bool) {
        try send(["cmd": "step", "action": action])
        let r = try recv()
        guard let ok = r["ok"] as? Bool, ok else {
            throw NSError(domain: "PythonEnvSession", code: 13, userInfo: [NSLocalizedDescriptionKey: r["error"] ?? r])
        }
        let obs = r["obs"] as? [Double]
        let mask = r["mask"] as? [Bool]
        let reward = r["reward"] as? Double ?? 0
        let terminated = r["done"] as? Bool ?? false
        let truncated = r["truncated"] as? Bool ?? false
        guard let o = obs, let m = mask else {
            throw NSError(domain: "PythonEnvSession", code: 14)
        }
        let episodeDone = terminated || truncated
        return (o.map { Float($0) }, m, Float(reward), episodeDone)
    }

    public func close() throws {
        try send(["cmd": "close"])
        _ = try? recv()
        stdinFH.closeFile()
    }

    public func debugState() throws -> [String: Any] {
        try send(["cmd": "debug_state"])
        let r = try recv()
        guard let ok = r["ok"] as? Bool, ok else {
            throw NSError(domain: "PythonEnvSession", code: 15, userInfo: [NSLocalizedDescriptionKey: r["error"] ?? r])
        }
        return r
    }
}

extension PythonEnvSession: TextSimEnvironment {}
