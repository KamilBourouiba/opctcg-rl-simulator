import Foundation

public struct NativeBoardChar {
    public var cardId: String
    public var power: Int
    public var rested: Bool
    public var justPlayed: Bool
    public var attachedDon: Int
    public var hasBlocker: Bool
    public var hasRush: Bool
    public var hasRushChar: Bool
    public var hasDoubleAttack: Bool
    public var hasUnblockable: Bool
    public var hasAttackActive: Bool
    public var activateMainUsed: Bool

    public init(card: SwiftCard, justPlayed: Bool) {
        cardId = card.cardId
        power = card.power
        rested = false
        self.justPlayed = justPlayed
        attachedDon = 0
        let t = card.loweredText
        hasBlocker = t.contains("[blocker]")
        hasRush = t.contains("[rush]") && !t.contains("rush: character")
        hasRushChar = t.contains("rush: character") || t.contains("[rush: character]")
        hasDoubleAttack = t.contains("[double attack]")
        hasUnblockable = t.contains("[unblockable]")
        hasAttackActive = t.contains("attack active character")
        activateMainUsed = false
    }
}

public struct NativePlayerState {
    public var deck: [String]
    public var hand: [String]
    public var leaderId: String?
    public var leaderPower: Int
    public var leaderRested: Bool
    public var leaderAttachedDon: Int
    public var leaderActivateMainUsed: Bool
    public var leaderOncePerTurnDrawUsed: Bool
    public var leaderOncePerTurnDefenseUsed: Bool
    public var leaderOpponentTurnLifeZeroUsed: Bool
    public var leaderPowerBonusTurn: Int
    public var board: [NativeBoardChar]
    public var lifeCards: [String]
    public var trash: [String]
    public var stageCardId: String?
    public var donActive: Int
    public var donRested: Int
    public var donDeck: Int
}
