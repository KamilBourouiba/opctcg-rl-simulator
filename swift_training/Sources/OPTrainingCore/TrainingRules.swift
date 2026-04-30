import Foundation

/// Réglages chemins (alignés conceptuellement sur ``config.yaml`` du dépôt Python).
public struct PathsConfig: Codable, Sendable {
    public var rulesPdf: String
    public var cardsCsv: String
    public var rulesCorpusOut: String

    public init(
        rulesPdf: String = "~/Downloads/rule_comprehensive.pdf",
        cardsCsv: String = "data/cards_tcgcsv.csv",
        rulesCorpusOut: String = "data/rules_corpus.txt"
    ) {
        self.rulesPdf = rulesPdf
        self.cardsCsv = cardsCsv
        self.rulesCorpusOut = rulesCorpusOut
    }
}

/// Hyperparamètres et dimensions — même vocabulaire que ``training:`` dans ``config.yaml``.
/// Pour cette cible Swift démo, ``syntheticNAct`` pilote un jeu de données synthétique MLX (pas le simulateur OPTCG).
public struct TrainingConfig: Codable, Sendable {
    public var totalSteps: Int
    public var rolloutLen: Int
    public var gamma: Float
    public var lam: Float
    public var lr: Float
    public var weightDecay: Float
    public var clipEpsilon: Float
    public var obsDim: Int
    public var hidden: Int
    public var nResBlocks: Int
    /// Nombre d’actions dans la démo MLX (régression / CE synthétiques).
    public var syntheticNAct: Int
    public var batchSize: Int
    public var seed: UInt64

    public init(
        totalSteps: Int = 4096,
        rolloutLen: Int = 512,
        gamma: Float = 0.99,
        lam: Float = 0.95,
        lr: Float = 2.5e-4,
        weightDecay: Float = 1e-4,
        clipEpsilon: Float = 0.2,
        obsDim: Int = 96,
        hidden: Int = 256,
        nResBlocks: Int = 2,
        syntheticNAct: Int = 64,
        batchSize: Int = 128,
        seed: UInt64 = 42
    ) {
        self.totalSteps = totalSteps
        self.rolloutLen = rolloutLen
        self.gamma = gamma
        self.lam = lam
        self.lr = lr
        self.weightDecay = weightDecay
        self.clipEpsilon = clipEpsilon
        self.obsDim = obsDim
        self.hidden = hidden
        self.nResBlocks = nResBlocks
        self.syntheticNAct = syntheticNAct
        self.batchSize = batchSize
        self.seed = seed
    }
}

public struct TrainingRules: Codable, Sendable {
    public var schemaVersion: Int
    public var paths: PathsConfig
    public var training: TrainingConfig

    public init(schemaVersion: Int = 1, paths: PathsConfig = PathsConfig(), training: TrainingConfig = TrainingConfig()) {
        self.schemaVersion = schemaVersion
        self.paths = paths
        self.training = training
    }
}

public enum TrainingRulesLoader {
    /// Charge un JSON UTF-8 (fichier ou URL ``file://``).
    public static func load(from url: URL) throws -> TrainingRules {
        let data = try Data(contentsOf: url)
        let dec = JSONDecoder()
        dec.keyDecodingStrategy = .convertFromSnakeCase
        return try dec.decode(TrainingRules.self, from: data)
    }

    /// Règles embarquées dans ``OPTrainingCore`` (copie du défaut du dépôt).
    public static func loadBundledDefault() throws -> TrainingRules {
        guard let url = Bundle.module.url(forResource: "default_training", withExtension: "json") else {
            throw CocoaError(.fileReadUnknown)
        }
        return try load(from: url)
    }
}
