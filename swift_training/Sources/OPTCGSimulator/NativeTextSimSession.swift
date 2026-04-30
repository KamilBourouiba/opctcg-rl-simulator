import Foundation
import OPTrainingCore

public final class NativeTextSimSession: TextSimEnvironment {
    private let simulator: NativeSimulator

    public init(
        repoRoot: URL,
        deck0: String,
        deck1: String?,
        cardsCsvRelativePath: String,
        columnMap: [String: String],
        obsDim: Int,
        seed: Int,
        shuffleDecks: Bool = true,
        selfPlay: Bool = true,
        enableTextEffects: Bool = true,
    ) throws {
        let csvURL = repoRoot.appendingPathComponent(cardsCsvRelativePath)
        let cards = try SwiftCSV.loadCards(path: csvURL, columnMap: columnMap)

        let d0URL = repoRoot.appendingPathComponent(deck0)
        let d1URL = repoRoot.appendingPathComponent(deck1 ?? deck0)

        let e0 = try SwiftDeckIO.parseDeckFile(at: d0URL)
        let e1 = try SwiftDeckIO.parseDeckFile(at: d1URL)
        var m0 = SwiftDeckIO.multiset(e0)
        var m1 = SwiftDeckIO.multiset(e1)

        let explicitLeader0 = try SwiftDeckIO.readLeaderDirective(at: d0URL)
        let explicitLeader1 = try SwiftDeckIO.readLeaderDirective(at: d1URL)
        let preferred0 = e0.map(\.0)
        let preferred1 = e1.map(\.0)
        let l0 = SwiftDeckIO.extractLeader(
            multiset: &m0,
            cards: cards,
            explicitLeaderId: explicitLeader0,
            preferredOrder: preferred0,
        )
        let l1 = SwiftDeckIO.extractLeader(
            multiset: &m1,
            cards: cards,
            explicitLeaderId: explicitLeader1,
            preferredOrder: preferred1,
        )

        let deck0Flat = SwiftDeckIO.flatDeck(m0)
        let deck1Flat = SwiftDeckIO.flatDeck(m1)

        simulator = NativeSimulator(
            cards: cards,
            deck0: deck0Flat,
            deck1: deck1Flat,
            leader0Id: l0,
            leader1Id: l1,
            config: NativeEnvConfig(
                obsDim: obsDim,
                enableTextEffects: enableTextEffects,
                shuffleDecks: shuffleDecks,
                selfPlay: selfPlay
            ),
            seed: seed,
        )
    }

    public func reset(seed: Int?) throws -> (obs: [Float], mask: [Bool]) {
        simulator.reset(seed: seed)
    }

    public func step(action: Int) throws -> (obs: [Float], mask: [Bool], reward: Float, done: Bool) {
        simulator.step(action: action)
    }

    public func close() throws {}

    public func replayState() -> NativeSimulator.ReplayState {
        simulator.exportReplayState()
    }

    public func actionDescription(_ action: Int) -> String {
        NativeSimulator.describeAction(action)
    }
}
