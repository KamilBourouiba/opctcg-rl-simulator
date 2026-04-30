import Foundation
import OPTrainingCore

public struct NativeEnvConfig {
    public var obsDim: Int
    public var startingHand: Int
    public var startingLife: Int
    public var leaderPower: Int
    public var maxBoard: Int
    public var maxDon: Int
    public var donPerTurn: Int
    public var enableTextEffects: Bool
    public var shuffleDecks: Bool
    public var selfPlay: Bool

    public init(
        obsDim: Int = GameConstants.defaultObsDim,
        startingHand: Int = 5,
        startingLife: Int = 5,
        leaderPower: Int = 5000,
        maxBoard: Int = 5,
        maxDon: Int = 10,
        donPerTurn: Int = 2,
        enableTextEffects: Bool = false,
        shuffleDecks: Bool = true,
        selfPlay: Bool = true,
    ) {
        self.obsDim = obsDim
        self.startingHand = startingHand
        self.startingLife = startingLife
        self.leaderPower = leaderPower
        self.maxBoard = maxBoard
        self.maxDon = maxDon
        self.donPerTurn = donPerTurn
        self.enableTextEffects = enableTextEffects
        self.shuffleDecks = shuffleDecks
        self.selfPlay = selfPlay
    }
}

public final class NativeSimulator {
    public struct ReplayBoardCard: Codable {
        public var id: String
        public var rested: Bool
        public var power: Int
        public var powerEffective: Int
        public var hasRush: Bool
        public var hasBlocker: Bool

        enum CodingKeys: String, CodingKey {
            case id
            case rested
            case power
            case powerEffective = "power_effective"
            case hasRush = "has_rush"
            case hasBlocker = "has_blocker"
        }
    }

    public struct ReplayPlayer: Codable {
        public var deckRemaining: Int
        public var life: Int
        public var lifeCards: [String]
        public var hand: [String]
        public var stageCardId: String?
        public var board: [ReplayBoardCard]
        public var trash: [String]
        public var leaderId: String?
        public var leaderPower: Int
        public var leaderPowerEffective: Int
        public var leaderRested: Bool
        public var leaderAttachedDon: Int
        public var donActive: Int
        public var donRested: Int
        public var donDeck: Int
        public var officialDonDeck: Int

        enum CodingKeys: String, CodingKey {
            case deckRemaining = "deck_remaining"
            case life
            case lifeCards = "life_cards"
            case hand
            case stageCardId = "stage_card_id"
            case board
            case trash
            case leaderId = "leader_id"
            case leaderPower = "leader_power"
            case leaderPowerEffective = "leader_power_effective"
            case leaderRested = "leader_rested"
            case leaderAttachedDon = "leader_attached_don"
            case donActive = "don_active"
            case donRested = "don_rested"
            case donDeck = "don_deck"
            case officialDonDeck = "official_don_deck"
        }
    }

    public struct ReplayState: Codable {
        public var phase: String
        public var done: Bool
        public var p0: ReplayPlayer
        public var p1: ReplayPlayer
        public var winner: Int?
    }

    private let cards: [String: SwiftCard]
    private let deck0Base: [String]
    private let deck1Base: [String]
    private let cfg: NativeEnvConfig
    private var rng: SplitMix64
    private let leader0Id: String?
    private let leader1Id: String?
    private var activePlayerIsP0 = true

    private var p0 = NativePlayerState(
        deck: [],
        hand: [],
        leaderId: nil,
        leaderPower: 5000,
        leaderRested: false,
        leaderAttachedDon: 0,
        leaderActivateMainUsed: false,
        leaderOncePerTurnDrawUsed: false,
        leaderOncePerTurnDefenseUsed: false,
        leaderOpponentTurnLifeZeroUsed: false,
        leaderPowerBonusTurn: 0,
        board: [],
        lifeCards: [],
        trash: [],
        stageCardId: nil,
        donActive: 0,
        donRested: 0,
        donDeck: 10,
    )
    private var p1 = NativePlayerState(
        deck: [],
        hand: [],
        leaderId: nil,
        leaderPower: 5000,
        leaderRested: false,
        leaderAttachedDon: 0,
        leaderActivateMainUsed: false,
        leaderOncePerTurnDrawUsed: false,
        leaderOncePerTurnDefenseUsed: false,
        leaderOpponentTurnLifeZeroUsed: false,
        leaderPowerBonusTurn: 0,
        board: [],
        lifeCards: [],
        trash: [],
        stageCardId: nil,
        donActive: 0,
        donRested: 0,
        donDeck: 10,
    )
    private var phase: SimPhase = .mulligan
    private var done = false
    private var turnsStarted = 0
    private var mainNonIdleActions = 0
    private var attacksMadeTurn = 0
    private var donAttachedTurn = 0
    private var blockerPending = false
    private var blockerAtkPower = 0
    private var blockerAtkDouble = false
    private var blockerAtkTarget = 0
    private var deferredLifeTriggers: [(cardId: String, defenderIsP0: Bool)] = []
    /// « Set up to N DON as active at the end of this tour » (sous-texte On Play / Main / …).
    private var scheduledEndOfThisTurnDonRefreshP0: [Int] = []
    private var scheduledEndOfThisTurnDonRefreshP1: [Int] = []
    /// Cible d’un malus de puissance « during this turn » sur un personnage adverse (simplifié : id stable).
    private var p0OppDebuffedCharId: String?
    private var p0OppDebuffAmount: Int = 0
    private var p1OppDebuffedCharId: String?
    private var p1OppDebuffAmount: Int = 0
    private var mulliganP0Done = false
    private var mulliganNeedP1 = false

    public init(
        cards: [String: SwiftCard],
        deck0: [String],
        deck1: [String]?,
        leader0Id: String?,
        leader1Id: String?,
        config: NativeEnvConfig,
        seed: Int,
    ) {
        self.cards = cards
        self.deck0Base = deck0
        self.deck1Base = deck1 ?? deck0
        self.leader0Id = leader0Id
        self.leader1Id = leader1Id
        self.cfg = config
        rng = SplitMix64(seed: UInt64(bitPattern: Int64(seed)))
    }

    public func reseed(_ seed: Int) {
        rng = SplitMix64(seed: UInt64(bitPattern: Int64(seed)))
    }

    public func reset(seed: Int?) -> (obs: [Float], mask: [Bool]) {
        if let s = seed {
            reseed(s)
        }
        done = false
        phase = .mulligan
        turnsStarted = 0
        p0 = freshPlayer(from: deck0Base, leaderId: leader0Id)
        p1 = freshPlayer(from: deck1Base, leaderId: leader1Id)
        mulliganP0Done = false
        mulliganNeedP1 = false
        scheduledEndOfThisTurnDonRefreshP0.removeAll()
        scheduledEndOfThisTurnDonRefreshP1.removeAll()
        p0OppDebuffedCharId = nil
        p0OppDebuffAmount = 0
        p1OppDebuffedCharId = nil
        p1OppDebuffAmount = 0
        resetTurnActionCounters()
        return (observation(), legalMask())
    }

    public func step(action: Int) -> (obs: [Float], mask: [Bool], reward: Float, done: Bool) {
        if done {
            return (observation(), legalMask(), 0, true)
        }
        var reward: Float = 0
        switch phase {
        case .mulligan:
            reward += stepMulligan(action: action)
        case .main:
            reward += applyMain(action: action)
        case .battle:
            reward += applyBattle(action: action)
        case .blocker:
            reward += stepBlocker(action: action)
        }
        if done {
            return (observation(), [Bool](repeating: false, count: SimConstants.actionSpaceSize), reward, true)
        }
        return (observation(), legalMask(), reward, false)
    }

    private func freshPlayer(from baseDeck: [String], leaderId: String?) -> NativePlayerState {
        var deck = baseDeck
        shuffle(&deck)
        let leader = leaderId.flatMap { cards[$0] }
        let lp = ((leader?.power ?? 0) > 0) ? (leader?.power ?? cfg.leaderPower) : cfg.leaderPower
        let life = ((leader?.life ?? 0) > 0) ? (leader?.life ?? cfg.startingLife) : cfg.startingLife
        let explicitDonDeck = ((leader?.donDeckSize ?? 0) > 0) ? (leader?.donDeckSize ?? cfg.maxDon) : cfg.maxDon
        let donDeck = leaderDonDeckSize(from: leader, fallback: explicitDonDeck)
        var p = NativePlayerState(
            deck: deck,
            hand: [],
            leaderId: leaderId,
            leaderPower: lp,
            leaderRested: false,
            leaderAttachedDon: 0,
            leaderActivateMainUsed: false,
            leaderOncePerTurnDrawUsed: false,
            leaderOncePerTurnDefenseUsed: false,
            leaderOpponentTurnLifeZeroUsed: false,
            leaderPowerBonusTurn: 0,
            board: [],
            lifeCards: [],
            trash: [],
            stageCardId: nil,
            donActive: 0,
            donRested: 0,
            donDeck: donDeck,
        )
        for _ in 0..<cfg.startingHand {
            if let c = drawCard(from: &p) { p.hand.append(c) }
        }
        for _ in 0..<life {
            if let c = drawCard(from: &p) { p.lifeCards.append(c) }
        }
        return p
    }

    private func drawCard(from p: inout NativePlayerState) -> String? {
        guard !p.deck.isEmpty else { return nil }
        return p.deck.removeFirst()
    }

    private func shuffle(_ a: inout [String]) {
        if !cfg.shuffleDecks { return }
        guard a.count > 1 else { return }
        for i in stride(from: a.count - 1, through: 1, by: -1) {
            let j = Int(rng.nextUInt64() % UInt64(i + 1))
            if i != j { a.swapAt(i, j) }
        }
    }

    private func startTurn(for p: inout NativePlayerState, firstTurn: Bool) {
        p.leaderRested = false
        p.leaderAttachedDon = 0
        p.leaderActivateMainUsed = false
        p.leaderOncePerTurnDrawUsed = false
        p.leaderOncePerTurnDefenseUsed = false
        p.leaderOpponentTurnLifeZeroUsed = false
        p.leaderPowerBonusTurn = 0
        p.board = p.board.map {
            var c = $0
            c.rested = false
            c.justPlayed = false
            c.attachedDon = 0
            c.activateMainUsed = false
            return c
        }
        p.donActive += p.donRested
        p.donRested = 0
        let addDon = firstTurn ? 1 : cfg.donPerTurn
        let gained = min(addDon, p.donDeck)
        p.donDeck -= gained
        p.donActive += gained
        if !firstTurn, let c = drawCard(from: &p) {
            p.hand.append(c)
        }
        resetTurnActionCounters()
    }

    private func resetTurnActionCounters() {
        mainNonIdleActions = 0
        attacksMadeTurn = 0
        donAttachedTurn = 0
    }

    private func redrawOpeningHand(_ p: inout NativePlayerState) {
        p.deck.append(contentsOf: p.hand)
        p.hand.removeAll(keepingCapacity: true)
        shuffle(&p.deck)
        for _ in 0..<cfg.startingHand {
            if let c = drawCard(from: &p) { p.hand.append(c) }
        }
    }

    private func stepMulligan(action: Int) -> Float {
        if !cfg.selfPlay {
            if action == SimConstants.mulliganTake {
                redrawOpeningHand(&p0)
            }
            var lowCost = 0
            for cid in p1.hand {
                if (cards[cid]?.cost ?? 99) <= 3 { lowCost += 1 }
            }
            if lowCost < 2 {
                redrawOpeningHand(&p1)
            }
            phase = .main
            activePlayerIsP0 = true
            startTurn(for: &p0, firstTurn: true)
            return 0
        }

        if !mulliganP0Done {
            if action == SimConstants.mulliganTake {
                redrawOpeningHand(&p0)
            }
            mulliganP0Done = true
            mulliganNeedP1 = true
            return 0
        }

        if mulliganNeedP1 {
            if action == SimConstants.mulliganTake {
                redrawOpeningHand(&p1)
            }
            mulliganNeedP1 = false
            mulliganP0Done = false
        }

        phase = .main
        activePlayerIsP0 = true
        startTurn(for: &p0, firstTurn: true)
        return 0
    }

    private func turnHasMeaningfulActivity() -> Bool {
        mainNonIdleActions > 0 || attacksMadeTurn > 0 || donAttachedTurn > 0
    }

    private func anyLegalMainCommit() -> Bool {
        if phase != .main { return false }
        for slot in 0..<min(7, p0.hand.count) {
            let cid = p0.hand[slot]
            guard let c = cards[cid] else { continue }
            if c.cost <= p0.donActive {
                return true
            }
        }
        if p0.donActive > 0 {
            for slot in 0..<SimConstants.mainAttachDonSlots {
                if slot == 0 || (slot - 1) < p0.board.count {
                    return true
                }
            }
        }
        for slot in 0..<min(SimConstants.mainActivateMainSlots, p0.board.count) {
            let cid = p0.board[slot].cardId
            if let c = cards[cid], c.hasActivateMain, !p0.board[slot].activateMainUsed, !p0.board[slot].rested {
                return true
            }
        }
        return canUseLeaderActivateMain()
    }

    private func anyLegalBattleCommit() -> Bool {
        if phase != .battle { return false }
        for atk in 0..<SimConstants.nAttackers {
            for tgt in 0..<SimConstants.nTargets where attackIsLegal(attacker: atk, target: tgt) {
                return true
            }
        }
        if p0.donActive > 0 {
            for slot in 0..<SimConstants.mainAttachDonSlots {
                if slot == 0 || (slot - 1) < p0.board.count {
                    return true
                }
            }
        }
        return false
    }

    private func attackIsLegal(attacker: Int, target: Int, forP1: Bool = false) -> Bool {
        let atk = forP1 ? p1 : p0
        let def = forP1 ? p0 : p1
        if attacker == 0 {
            if atk.leaderRested { return false }
            if !forP1, turnsStarted <= 1 { return false }
            if forP1, turnsStarted <= 2 { return false }
        } else {
            let aIdx = attacker - 1
            guard aIdx >= 0, aIdx < atk.board.count, canAttack(atk.board[aIdx]) else { return false }
            if !forP1, turnsStarted <= 1 { return false }
            if forP1, turnsStarted <= 2 { return false }
            if atk.board[aIdx].hasRushChar && atk.board[aIdx].justPlayed && target == 0 { return false }
        }
        if target == 0 { return true }
        let tIdx = target - 1
        if !(tIdx >= 0 && tIdx < def.board.count) { return false }
        if def.board[tIdx].rested { return true }
        // Simplified: only cards with explicit active-character permission can hit active targets.
        if attacker == 0 { return false }
        let aIdx = attacker - 1
        return atk.board[aIdx].hasAttackActive
    }

    private func endOurTurnAndPlayOpponent() -> Float {
        if cfg.enableTextEffects {
            for n in scheduledEndOfThisTurnDonRefreshP0 { refreshDon(ownerIsP0: true, count: n) }
            scheduledEndOfThisTurnDonRefreshP0.removeAll()
            applyEndOfYourTurnLabeledEffectsFromBoard(ownerIsP0: true)
        }
        p1OppDebuffedCharId = nil
        p1OppDebuffAmount = 0
        p0.donRested += p0.donActive
        p0.donActive = 0
        p0.leaderAttachedDon = 0
        p0.board = p0.board.map {
            var c = $0
            c.attachedDon = 0
            return c
        }

        turnsStarted += 1
        activePlayerIsP0 = false
        startTurn(for: &p1, firstTurn: false)
        opponentMainAndBattle()
        if done { return 0 }

        p1.donRested += p1.donActive
        p1.donActive = 0
        p1.leaderAttachedDon = 0
        p1.board = p1.board.map {
            var c = $0
            c.attachedDon = 0
            return c
        }

        turnsStarted += 1
        activePlayerIsP0 = true
        startTurn(for: &p0, firstTurn: false)
        phase = .main
        return 0
    }

    private func applyMain(action: Int) -> Float {
        if action == SimConstants.mainEndAction {
            if !(turnHasMeaningfulActivity() || !anyLegalMainCommit()) {
                return -0.08
            }
            phase = .battle
            return 0
        }

        if (0...6).contains(action) {
            return playFromHand(slot: action)
        }

        if action >= SimConstants.mainAttachDonBase && action < SimConstants.mainActivateMainBase {
            let rel = action - SimConstants.mainAttachDonBase
            let slot = rel / SimConstants.mainAttachDonMax
            let amount = (rel % SimConstants.mainAttachDonMax) + 1
            return attachDon(slot: slot, amount: amount)
        }

        if action >= SimConstants.mainActivateMainBase && action < SimConstants.mainActivateMainLeader {
            let slot = action - SimConstants.mainActivateMainBase
            return activateMainOnBoard(slot: slot)
        }

        if action == SimConstants.mainActivateMainLeader {
            return activateMainLeader()
        }
        return -0.01
    }

    private func playFromHand(slot: Int) -> Float {
        guard slot >= 0, slot < p0.hand.count else { return -0.02 }
        let cid = p0.hand[slot]
        let card = cards[cid] ?? SwiftCard(
            cardId: cid,
            name: cid,
            cost: 0,
            power: 0,
            counter: 0,
            color: "",
            cardText: "",
            cardType: "character",
        )
        let loweredType = card.loweredType
        let mainDonRestCost: Int
        if loweredType.contains("event"), cfg.enableTextEffects,
           let mainSeg = TextTimingSegments.extractBodyAfterLabel(card.cardText, timing: "main")
        {
            mainDonRestCost = parseDonRestCost(mainSeg.lowercased())
        } else {
            mainDonRestCost = 0
        }

        let totalNeed = card.cost + mainDonRestCost
        guard totalNeed <= p0.donActive else { return -0.02 }
        p0.donActive -= card.cost
        p0.donRested += card.cost
        if mainDonRestCost > 0 {
            let paid = payDonRest(on: &p0, amount: mainDonRestCost)
            if paid < mainDonRestCost { return -0.02 }
        }
        p0.hand.remove(at: slot)
        if loweredType.contains("character"), p0.board.count < cfg.maxBoard {
            p0.board.append(NativeBoardChar(card: card, justPlayed: true))
            if cfg.enableTextEffects {
                applyTimingTextEffects(
                    for: card,
                    ownerIsP0: true,
                    timing: "on_play",
                    sourceIsLeader: false,
                    sourceBoardIndex: p0.board.count - 1
                )
            }
            mainNonIdleActions += 1
            return 0.05
        }
        if loweredType.contains("stage") {
            if let old = p0.stageCardId { p0.trash.append(old) }
            p0.stageCardId = card.cardId
            if cfg.enableTextEffects {
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "on_play")
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "main")
            }
            mainNonIdleActions += 1
            return 0.03
        }
        if loweredType.contains("event"), cfg.enableTextEffects {
            applyTimingTextEffects(for: card, ownerIsP0: true, timing: "main")
        }
        mainNonIdleActions += 1
        p0.trash.append(cid)
        return 0.01
    }

    private func attachDon(slot: Int, amount: Int) -> Float {
        guard amount > 0, amount <= p0.donActive else { return -0.02 }
        if slot == 0 {
            p0.leaderAttachedDon += amount
            p0.donActive -= amount
            p0.donRested += amount
            mainNonIdleActions += 1
            donAttachedTurn += amount
            return 0.02
        }
        let idx = slot - 1
        guard idx >= 0, idx < p0.board.count else { return -0.02 }
        p0.board[idx].attachedDon += amount
        p0.donActive -= amount
        p0.donRested += amount
        mainNonIdleActions += 1
        donAttachedTurn += amount
        return 0.02
    }

    private func activateMainOnBoard(slot: Int) -> Float {
        guard slot >= 0, slot < p0.board.count else { return -0.02 }
        let cid = p0.board[slot].cardId
        guard let card = cards[cid], card.hasActivateMain, !p0.board[slot].activateMainUsed else { return -0.02 }
        let cost = parseDonMinusCost(card.loweredText)
        if cost > 0 {
            let paid = payDonMinus(on: &p0, amount: cost)
            if paid < cost { return -0.02 }
        }
        p0.board[slot].activateMainUsed = true
        p0.board[slot].rested = true
        if cfg.enableTextEffects {
            applyTimingTextEffects(for: card, ownerIsP0: true, timing: "activate_main")
        }
        mainNonIdleActions += 1
        return 0.04
    }

    private func activateMainLeader() -> Float {
        guard canUseLeaderActivateMain() else { return -0.02 }
        if cfg.enableTextEffects, let lid = p0.leaderId, let lc = cards[lid] {
            let txt = lc.loweredText
            if txt.contains("second turn or later"), ((turnsStarted + 1) / 2) < 2 {
                return -0.02
            }
            let cost = parseDonMinusCost(txt)
            if cost > 0 {
                let paid = payDonMinus(on: &p0, amount: cost)
                if paid < cost { return -0.02 }
            }
            applyTimingTextEffects(for: lc, ownerIsP0: true, timing: "activate_main", sourceIsLeader: true)
        }
        p0.leaderActivateMainUsed = true
        mainNonIdleActions += 1
        return 0.04
    }

    private func canUseLeaderActivateMain() -> Bool {
        guard !p0.leaderActivateMainUsed else { return false }
        guard let lid = p0.leaderId, let lc = cards[lid], lc.hasActivateMain else { return false }
        let txt = lc.loweredText
        if txt.contains("second turn or later"), ((turnsStarted + 1) / 2) < 2 {
            return false
        }
        let cost = parseDonMinusCost(txt)
        if cost <= 0 { return true }
        let donTotal = p0.donActive + p0.leaderAttachedDon + p0.board.reduce(0) { $0 + $1.attachedDon }
        return donTotal >= cost
    }

    private func canAttack(_ ch: NativeBoardChar) -> Bool {
        if ch.rested { return false }
        if ch.justPlayed && !ch.hasRush && !ch.hasRushChar { return false }
        return true
    }

    private func attackPowerLeader(_ p: NativePlayerState, includeAttachedDonBonus: Bool = true) -> Int {
        p.leaderPower + p.leaderPowerBonusTurn + (includeAttachedDonBonus ? p.leaderAttachedDon * 1000 : 0)
    }

    private func attackPowerChar(_ c: NativeBoardChar, includeAttachedDonBonus: Bool = true) -> Int {
        c.power + (includeAttachedDonBonus ? c.attachedDon * 1000 : 0)
    }

    /// Puissance de défense d’un personnage (malus « -N power during this turn » sur cible adverse).
    private func defensePowerChar(_ c: NativeBoardChar, defenderIsP0: Bool, includeAttachedDonBonus: Bool) -> Int {
        var p = attackPowerChar(c, includeAttachedDonBonus: includeAttachedDonBonus)
        if defenderIsP0, let id = p0OppDebuffedCharId, id == c.cardId {
            p = max(0, p - p0OppDebuffAmount)
        }
        if !defenderIsP0, let id = p1OppDebuffedCharId, id == c.cardId {
            p = max(0, p - p1OppDebuffAmount)
        }
        return p
    }

    private func applyBattle(action: Int) -> Float {
        if action == SimConstants.battlePassAction {
            if !(turnHasMeaningfulActivity() || !anyLegalBattleCommit()) {
                return -0.08
            }
            return endOurTurnAndPlayOpponent()
        }
        guard action >= SimConstants.battleAttackBase, action < SimConstants.mainAttachDonBase else { return -0.02 }
        let rel = action - SimConstants.battleAttackBase
        let attacker = rel / SimConstants.nTargets
        let target = rel % SimConstants.nTargets
        return resolveOurAttack(attacker: attacker, target: target)
    }

    private func resolveOurAttack(attacker: Int, target: Int) -> Float {
        guard attackIsLegal(attacker: attacker, target: target) else { return -0.05 }
        let atkPower: Int
        if attacker == 0 {
            atkPower = attackPowerLeader(p0, includeAttachedDonBonus: true)
            p0.leaderRested = true
            if cfg.enableTextEffects, let lid = p0.leaderId, let card = cards[lid] {
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "when_attacking", sourceIsLeader: true)
            }
        } else {
            let aIdx = attacker - 1
            atkPower = attackPowerChar(p0.board[aIdx], includeAttachedDonBonus: true)
            p0.board[aIdx].rested = true
            if cfg.enableTextEffects, let card = cards[p0.board[aIdx].cardId] {
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "when_attacking", sourceIsLeader: false, sourceBoardIndex: aIdx)
            }
        }
        attacksMadeTurn += 1

        if target == 0 {
            let defended = applyCounterFromHand(
                defender: &p1,
                attackPower: atkPower,
                targetIsLeader: true,
                includeAttachedDonBonus: false,
                defenderIsP0: false
            )
            if atkPower >= defended {
                let r = hitLeader(attackPower: atkPower, defender: &p1, defenderIsP0: false)
                flushDeferredLifeTriggers()
                return r
            }
            return -0.01
        }
        let dIdx = target - 1
        guard dIdx >= 0, dIdx < p1.board.count else { return -0.01 }
        let defPower = applyCounterFromHand(
            defender: &p1,
            attackPower: atkPower,
            targetIsLeader: false,
            characterIndex: dIdx,
            includeAttachedDonBonus: false,
            defenderIsP0: false
        )
        if atkPower >= defPower {
            let def = p1.board[dIdx]
            if p1OppDebuffedCharId == def.cardId {
                p1OppDebuffedCharId = nil
                p1OppDebuffAmount = 0
            }
            if cfg.enableTextEffects, let card = cards[def.cardId] {
                applyTimingTextEffects(for: card, ownerIsP0: false, timing: "on_ko", sourceIsLeader: false, sourceBoardIndex: dIdx)
            }
            p1.trash.append(def.cardId)
            p1.board.remove(at: dIdx)
            return 0.08
        }
        return 0.01
    }

    private func hitLeader(attackPower: Int, defender: inout NativePlayerState, defenderIsP0: Bool) -> Float {
        let defPower = attackPowerLeader(defender)
        guard attackPower >= defPower else { return -0.01 }
        if !defender.lifeCards.isEmpty {
            let lifeCard = defender.lifeCards.removeLast()
            defender.hand.append(lifeCard)
            if cfg.enableTextEffects {
                deferredLifeTriggers.append((cardId: lifeCard, defenderIsP0: defenderIsP0))
                triggerLeaderAutoDrawForActivePlayer()
                triggerLeaderOpponentTurnLifeZeroIfNeeded(defender: &defender, defenderIsP0: defenderIsP0)
            }
            return 0.12
        }
        done = true
        return 1.0
    }

    private func opponentMainAndBattle() {
        while p1.donActive > 0 {
            guard let i = p1.hand.indices.first(where: { idx in
                let cid = p1.hand[idx]
                guard let c = cards[cid] else { return false }
                var extra = 0
                if cfg.enableTextEffects, c.loweredType.contains("event"),
                   let mainSeg = TextTimingSegments.extractBodyAfterLabel(c.cardText, timing: "main")
                {
                    extra = parseDonRestCost(mainSeg.lowercased())
                }
                return (c.cost + extra) <= p1.donActive
            }) else { break }
            let cid = p1.hand[i]
            let card = cards[cid] ?? SwiftCard(
                cardId: cid,
                name: cid,
                cost: 0,
                power: 0,
                counter: 0,
                color: "",
                cardText: "",
                cardType: "character",
            )
            var extraMainRest = 0
            if cfg.enableTextEffects, card.loweredType.contains("event"),
               let mainSeg = TextTimingSegments.extractBodyAfterLabel(card.cardText, timing: "main")
            {
                extraMainRest = parseDonRestCost(mainSeg.lowercased())
            }
            p1.donActive -= card.cost
            p1.donRested += card.cost
            if extraMainRest > 0 {
                let paid = payDonRest(on: &p1, amount: extraMainRest)
                if paid < extraMainRest { break }
            }
            p1.hand.remove(at: i)
            let loweredType = card.loweredType
            if loweredType.contains("character"), p1.board.count < cfg.maxBoard {
                p1.board.append(NativeBoardChar(card: card, justPlayed: true))
                if cfg.enableTextEffects {
                    applyTimingTextEffects(
                        for: card,
                        ownerIsP0: false,
                        timing: "on_play",
                        sourceIsLeader: false,
                        sourceBoardIndex: p1.board.count - 1
                    )
                }
            } else if loweredType.contains("stage") {
                if let old = p1.stageCardId { p1.trash.append(old) }
                p1.stageCardId = card.cardId
                if cfg.enableTextEffects {
                    applyTimingTextEffects(for: card, ownerIsP0: false, timing: "on_play")
                    applyTimingTextEffects(for: card, ownerIsP0: false, timing: "main")
                }
            } else if loweredType.contains("event"), cfg.enableTextEffects {
                applyTimingTextEffects(for: card, ownerIsP0: false, timing: "main")
                p1.trash.append(cid)
            } else {
                p1.trash.append(cid)
            }
        }
        if p1.donActive > 0, !p1.board.isEmpty {
            let target = Int(rng.nextUInt64() % UInt64(p1.board.count + 1))
            if target == 0 {
                p1.leaderAttachedDon += 1
            } else {
                p1.board[target - 1].attachedDon += 1
            }
            p1.donActive -= 1
            p1.donRested += 1
        } else if p1.donActive > 0 {
            p1.leaderAttachedDon += 1
            p1.donActive -= 1
            p1.donRested += 1
        }

        var attacksUsed = 0
        while !done && attacksUsed < 6 {
            var legal: [(Int, Int)] = []
            for a in 0..<SimConstants.nAttackers {
                for t in 0..<SimConstants.nTargets where attackIsLegal(attacker: a, target: t, forP1: true) {
                    legal.append((a, t))
                }
            }
            if legal.isEmpty { break }
            let pick = legal[Int(rng.nextUInt64() % UInt64(legal.count))]
            if cfg.enableTextEffects { applyLeaderDefensiveBoostIfAny(defender: &p0) }
            if tryPauseBlockerOrResolveP1Attack(attacker: pick.0, target: pick.1) {
                return
            }
            attacksUsed += 1
        }
        if cfg.enableTextEffects {
            for n in scheduledEndOfThisTurnDonRefreshP1 { refreshDon(ownerIsP0: false, count: n) }
            scheduledEndOfThisTurnDonRefreshP1.removeAll()
            applyEndOfYourTurnLabeledEffectsFromBoard(ownerIsP0: false)
        }
        p0OppDebuffedCharId = nil
        p0OppDebuffAmount = 0
        if p0.lifeCards.isEmpty && p0.leaderPower <= 0 {
            done = true
        }
    }

    private func applyEndOfYourTurnLabeledEffectsFromBoard(ownerIsP0: Bool) {
        guard cfg.enableTextEffects else { return }
        let st = ownerIsP0 ? p0 : p1
        if let lid = st.leaderId, let c = cards[lid], c.loweredText.contains("[end of your turn]") {
            applyTimingTextEffects(for: c, ownerIsP0: ownerIsP0, timing: "end_of_your_turn", sourceIsLeader: true)
        }
        for (i, b) in st.board.enumerated() {
            if let c = cards[b.cardId], c.loweredText.contains("[end of your turn]") {
                applyTimingTextEffects(for: c, ownerIsP0: ownerIsP0, timing: "end_of_your_turn", sourceBoardIndex: i)
            }
        }
        if let sid = st.stageCardId, let c = cards[sid], c.loweredText.contains("[end of your turn]") {
            applyTimingTextEffects(for: c, ownerIsP0: ownerIsP0, timing: "end_of_your_turn")
        }
    }

    private func legalMask() -> [Bool] {
        if done {
            return [Bool](repeating: false, count: SimConstants.actionSpaceSize)
        }
        var m = [Bool](repeating: false, count: SimConstants.actionSpaceSize)
        switch phase {
        case .mulligan:
            m[SimConstants.mulliganKeep] = true
            m[SimConstants.mulliganTake] = true
        case .main:
            m[SimConstants.mainEndAction] = turnHasMeaningfulActivity() || !anyLegalMainCommit()
            for a in 0...6 {
                let slot = a
                if slot < p0.hand.count {
                    let cid = p0.hand[slot]
                    guard let c = cards[cid] else { continue }
                    var extra = 0
                    if c.loweredType.contains("event"), cfg.enableTextEffects,
                       let mainSeg = TextTimingSegments.extractBodyAfterLabel(c.cardText, timing: "main")
                    {
                        extra = parseDonRestCost(mainSeg.lowercased())
                    }
                    if (c.cost + extra) <= p0.donActive {
                        m[a] = true
                    }
                }
            }
            for slot in 0..<SimConstants.mainAttachDonSlots {
                let slotPlayable = slot == 0 || (slot - 1) < p0.board.count
                if !slotPlayable { continue }
                let maxAttach = min(SimConstants.mainAttachDonMax, p0.donActive)
                if maxAttach < 1 { continue }
                for amount in 1...maxAttach {
                    let a = SimConstants.mainAttachDonBase + slot * SimConstants.mainAttachDonMax + (amount - 1)
                    m[a] = true
                }
            }
            for slot in 0..<min(SimConstants.mainActivateMainSlots, p0.board.count) {
                let cid = p0.board[slot].cardId
                if let c = cards[cid], c.hasActivateMain, !p0.board[slot].activateMainUsed {
                    let need = parseDonMinusCost(c.loweredText)
                    let available = p0.donActive + p0.leaderAttachedDon + p0.board.reduce(0) { $0 + $1.attachedDon }
                    if need > available { continue }
                    m[SimConstants.mainActivateMainBase + slot] = true
                }
            }
            if canUseLeaderActivateMain() {
                m[SimConstants.mainActivateMainLeader] = true
            }
        case .battle:
            m[SimConstants.battlePassAction] = turnHasMeaningfulActivity() || !anyLegalBattleCommit()
            for attacker in 0..<SimConstants.nAttackers {
                for target in 0..<SimConstants.nTargets {
                    if !attackIsLegal(attacker: attacker, target: target, forP1: false) { continue }
                    let a = SimConstants.battleAttackBase + attacker * SimConstants.nTargets + target
                    m[a] = true
                }
            }
            for slot in 0..<SimConstants.mainAttachDonSlots {
                let slotPlayable = slot == 0 || (slot - 1) < p0.board.count
                if !slotPlayable { continue }
                let maxAttach = min(SimConstants.mainAttachDonMax, p0.donActive)
                if maxAttach < 1 { continue }
                for amount in 1...maxAttach {
                    let a = SimConstants.mainAttachDonBase + slot * SimConstants.mainAttachDonMax + (amount - 1)
                    m[a] = true
                }
            }
        case .blocker:
            m[SimConstants.blockerPass] = true
            for i in 0..<min(5, p0.board.count) {
                if p0.board[i].hasBlocker && !p0.board[i].rested {
                    m[SimConstants.blockerSlotBase + i] = true
                }
            }
        }
        return m
    }

    private func tryPauseBlockerOrResolveP1Attack(attacker: Int, target: Int) -> Bool {
        let atkPower: Int
        let atkUnblockable: Bool
        if attacker == 0 {
            atkPower = attackPowerLeader(p1, includeAttachedDonBonus: true)
            p1.leaderRested = true
            if cfg.enableTextEffects, let lid = p1.leaderId, let card = cards[lid] {
                applyTimingTextEffects(for: card, ownerIsP0: false, timing: "when_attacking", sourceIsLeader: true)
            }
            blockerAtkDouble = false
            if let lid = p1.leaderId, let c = cards[lid] {
                atkUnblockable = c.loweredText.contains("[unblockable]")
            } else {
                atkUnblockable = false
            }
        } else {
            let idx = attacker - 1
            atkPower = attackPowerChar(p1.board[idx], includeAttachedDonBonus: true)
            p1.board[idx].rested = true
            if cfg.enableTextEffects, let card = cards[p1.board[idx].cardId] {
                applyTimingTextEffects(for: card, ownerIsP0: false, timing: "when_attacking", sourceIsLeader: false, sourceBoardIndex: idx)
            }
            blockerAtkDouble = p1.board[idx].hasDoubleAttack
            atkUnblockable = p1.board[idx].hasUnblockable
        }
        attacksMadeTurn += 1
        let blockerExists = p0.board.contains(where: { $0.hasBlocker && !$0.rested }) && !atkUnblockable
        if blockerExists {
            blockerPending = true
            blockerAtkPower = atkPower
            blockerAtkTarget = target
            phase = .blocker
            return true
        }
        _ = resolveP1AttackDamage(target: target, attackPower: atkPower, isDouble: blockerAtkDouble)
        return false
    }

    private func resolveP1AttackDamage(target: Int, attackPower: Int, isDouble: Bool) -> Float {
        if target == 0 {
            let defended = applyCounterFromHand(
                defender: &p0,
                attackPower: attackPower,
                targetIsLeader: true,
                includeAttachedDonBonus: false,
                defenderIsP0: true
            )
            if attackPower < defended { return -0.01 }
            let n = isDouble ? 2 : 1
            var reward: Float = 0
            for _ in 0..<n where !done {
                reward += hitLeader(attackPower: attackPower, defender: &p0, defenderIsP0: true)
                flushDeferredLifeTriggers()
            }
            return reward
        }
        let idx = target - 1
        guard idx >= 0, idx < p0.board.count else { return 0 }
        let defPower = applyCounterFromHand(
            defender: &p0,
            attackPower: attackPower,
            targetIsLeader: false,
            characterIndex: idx,
            includeAttachedDonBonus: false,
            defenderIsP0: true
        )
        if attackPower >= defPower {
            let def = p0.board[idx]
            if p0OppDebuffedCharId == def.cardId {
                p0OppDebuffedCharId = nil
                p0OppDebuffAmount = 0
            }
            if cfg.enableTextEffects, let card = cards[def.cardId] {
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "on_ko", sourceIsLeader: false, sourceBoardIndex: idx)
            }
            p0.trash.append(def.cardId)
            p0.board.remove(at: idx)
        }
        return 0
    }

    private func stepBlocker(action: Int) -> Float {
        guard blockerPending else {
            phase = .battle
            return -0.05
        }
        blockerPending = false
        phase = .battle
        if action == SimConstants.blockerPass {
            return resolveP1AttackDamage(target: blockerAtkTarget, attackPower: blockerAtkPower, isDouble: blockerAtkDouble)
        }
        let idx = action - SimConstants.blockerSlotBase
        guard idx >= 0, idx < p0.board.count, p0.board[idx].hasBlocker, !p0.board[idx].rested else {
            return resolveP1AttackDamage(target: blockerAtkTarget, attackPower: blockerAtkPower, isDouble: blockerAtkDouble)
        }
        p0.board[idx].rested = true
        let defPower = defensePowerChar(p0.board[idx], defenderIsP0: true, includeAttachedDonBonus: false)
        let defended = applyCounterFromHand(
            defender: &p0,
            attackPower: blockerAtkPower,
            targetIsLeader: false,
            characterIndex: idx,
            includeAttachedDonBonus: false,
            defenderIsP0: true
        )
        if blockerAtkPower >= max(defPower, defended) {
            let killed = p0.board.remove(at: idx)
            if cfg.enableTextEffects, let card = cards[killed.cardId] {
                applyTimingTextEffects(for: card, ownerIsP0: true, timing: "on_ko", sourceIsLeader: false, sourceBoardIndex: idx)
            }
            p0.trash.append(killed.cardId)
        }
        return 0
    }

    private func applyCounterFromHand(
        defender: inout NativePlayerState,
        attackPower: Int,
        targetIsLeader: Bool,
        characterIndex: Int? = nil,
        includeAttachedDonBonus: Bool = true,
        defenderIsP0: Bool
    ) -> Int {
        var defense: Int
        if targetIsLeader {
            defense = attackPowerLeader(defender, includeAttachedDonBonus: includeAttachedDonBonus)
        } else if let i = characterIndex, i >= 0, i < defender.board.count {
            defense = defensePowerChar(defender.board[i], defenderIsP0: defenderIsP0, includeAttachedDonBonus: includeAttachedDonBonus)
        } else {
            return 0
        }
        if defense >= attackPower { return defense }

        // Greedy: use highest counter cards first (Python has richer stack rules).
        while defense < attackPower {
            var bestIdx: Int?
            var bestCtr = 0
            for i in defender.hand.indices {
                let cid = defender.hand[i]
                let ctr = cards[cid]?.counter ?? 0
                if ctr > bestCtr {
                    bestCtr = ctr
                    bestIdx = i
                }
            }
            guard let bi = bestIdx, bestCtr > 0 else { break }
            let used = defender.hand.remove(at: bi)
            defender.trash.append(used)
            defense += bestCtr
        }
        return defense
    }

    private func parseDonMinusCost(_ text: String) -> Int {
        guard let r = text.range(of: "don!! -") else { return 0 }
        let sub = text[r.upperBound...]
        var digits = ""
        for ch in sub {
            if ch.isNumber { digits.append(ch) }
            else if !digits.isEmpty { break }
        }
        return Int(digits) ?? 0
    }

    private func parseDonRestCost(_ text: String) -> Int {
        guard let re = try? NSRegularExpression(pattern: #"don!!\s*(\d+)\s*:"#, options: [.caseInsensitive]) else { return 0 }
        let ns = text as NSString
        let r = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: text, options: [], range: r), m.numberOfRanges >= 2 else { return 0 }
        return Int(ns.substring(with: m.range(at: 1))) ?? 0
    }

    private func payDonRest(on p: inout NativePlayerState, amount: Int) -> Int {
        guard amount > 0 else { return 0 }
        let paid = min(amount, p.donActive)
        p.donActive -= paid
        p.donRested += paid
        return paid
    }

    private func payDonMinus(on p: inout NativePlayerState, amount: Int) -> Int {
        var rem = amount
        var paid = 0
        let a = min(rem, p.donActive)
        p.donActive -= a; p.donDeck += a; rem -= a; paid += a
        if rem > 0 {
            let l = min(rem, p.leaderAttachedDon)
            p.leaderAttachedDon -= l; p.donDeck += l; rem -= l; paid += l
        }
        if rem > 0 {
            for i in p.board.indices where rem > 0 {
                let t = min(rem, p.board[i].attachedDon)
                p.board[i].attachedDon -= t
                p.donDeck += t
                rem -= t
                paid += t
            }
        }
        return paid
    }

    private func applyTimingTextEffects(
        for card: SwiftCard,
        ownerIsP0: Bool,
        timing: String,
        sourceIsLeader: Bool? = nil,
        sourceBoardIndex: Int? = nil
    ) {
        let full = card.cardText
        let blob: String
        if timing == "activate_main" {
            guard let b = TextTimingSegments.extractActivateMainEffectText(full), !b.isEmpty else { return }
            blob = b
        } else if timing == "on_play" {
            let w = TextTimingSegments.extractOnPlayEffectWindow(full)
            if w.isEmpty { return }
            blob = w
        } else {
            guard let b = TextTimingSegments.extractBodyAfterLabel(full, timing: timing), !b.isEmpty else { return }
            blob = b
        }
        var txt = blob.lowercased()
        if txt.isEmpty { return }

        let donReq = TextTimingSegments.donXRequirementBeforeTimingLabel(full, timing: timing)
        if donReq > 0, !meetsDonXRequirement(
            ownerIsP0: ownerIsP0,
            required: donReq,
            sourceIsLeader: sourceIsLeader,
            sourceBoardIndex: sourceBoardIndex,
            sourceCardId: card.cardId
        ) {
            return
        }

        // DON!! N: coût en repos (utile notamment sur certains [Activate: Main]).
        if timing != "main" {
            let restCost = parseDonRestCost(txt)
            if restCost > 0 {
                if ownerIsP0 {
                    let paid = payDonRest(on: &p0, amount: restCost)
                    if paid < restCost { return }
                } else {
                    let paid = payDonRest(on: &p1, amount: restCost)
                    if paid < restCost { return }
                }
            }
        }

        // DON!! -X: paiement global du segment (hors activate_main déjà payé à l'action).
        if timing != "activate_main" {
            let segmentCost = parseDonMinusCost(txt)
            if segmentCost > 0 {
                if ownerIsP0 {
                    let paid = payDonMinus(on: &p0, amount: segmentCost)
                    if paid < segmentCost { return }
                } else {
                    let paid = payDonMinus(on: &p1, amount: segmentCost)
                    if paid < segmentCost { return }
                }
            }
        }

        // Normalize optional wording for easier regex matching.
        txt = txt.replacingOccurrences(of: "you may ", with: "")
        guard var work = preprocessConditionalText(txt, ownerIsP0: ownerIsP0) else { return }

        // Deck : puis effets multi-parties (alignés sur ``_RULES`` + CR 2-8-3 : on consomme chaque morceau pour le suivant).
        applyDeckWindowEffectsLoop(work: &work, ownerIsP0: ownerIsP0)
        applyCompositeHandDeckEffectsLoop(work: &work, ownerIsP0: ownerIsP0)

        if let n = firstIntMatch(
            in: work,
            pattern: #"trash up to (\d+).{0,200}?opponent['']?s life"#
        ),
            n > 0
        {
            trashOpponentLife(ownerIsP0: ownerIsP0, count: min(5, n))
        } else if work.contains("trash 1 of your opponent's life") { trashOpponentLife(ownerIsP0: ownerIsP0, count: 1) }

        // Du dessus de la vie adverse → main adverse (propriétaire de la zone) : Nami (3+ vies requis), Kuma (sans condition sur [on k.o.], etc.)
        if regexContains(
            work,
            #"add up to \d+ (?:card|cards) from the top of your opponent's life.{0,120}?to the owner's hand"#
        ) {
            let needOppLife: Int? = firstIntMatch(
                in: work,
                pattern: #"if your opponent has (\d+) or more life cards?"#
            )
            let canSteal: Bool
            if let need = needOppLife {
                canSteal = (ownerIsP0 ? p1.lifeCards.count : p0.lifeCards.count) >= need
            } else {
                canSteal = true
            }
            if canSteal {
                if ownerIsP0, !p1.lifeCards.isEmpty {
                    p1.hand.append(p1.lifeCards.removeLast())
                } else if !ownerIsP0, !p0.lifeCards.isEmpty {
                    p0.hand.append(p0.lifeCards.removeLast())
                }
            }
        }

        if let n = firstIntMatch(in: work, pattern: #"draw\s+(\d+)\s+cards?"#), n > 0 {
            drawCards(ownerIsP0: ownerIsP0, count: n)
        } else if regexContains(work, #"draw\s+(?:a|one|1)\s+card"#) {
            drawCards(ownerIsP0: ownerIsP0, count: 1)
        }

        if let n = firstIntMatch(in: work, pattern: #"discard\s+(\d+)\s+cards?"#), n > 0 {
            discardSelf(ownerIsP0: ownerIsP0, count: n)
        } else if regexContains(work, #"discard\s+(?:a|one|1)\s+card"#) {
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
        }
        if let n = firstIntMatch(in: work, pattern: #"trash\s+(\d+)\s+cards?\s+from\s+your\s+hand"#), n > 0 {
            discardSelf(ownerIsP0: ownerIsP0, count: n)
        } else if regexContains(work, #"trash\s+(?:a|one|1)\s+card\s+from\s+your\s+hand"#) {
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
        }

        if let n = firstIntMatch(in: work, pattern: #"your opponent trashes?\s+(\d+)\s+cards?\s+from\s+their\s+hand"#), n > 0 {
            discardOpponent(ownerIsP0: ownerIsP0, count: n)
        } else if work.contains("your opponent trashes 1 card from their hand")
            || work.contains("trash 1 card from your opponent's hand")
        {
            discardOpponent(ownerIsP0: ownerIsP0, count: 1)
        }

        // OP15 Enel (violet) : 1) +1 actif, +4 « additionnelles … and rest them » 2) transfert des DON **reposés** (coût) → perso. Pas « piocher 4 de plus à la toute fin ».
        if let n = firstIntMatch(in: work, pattern: #"add up to (\d+).*?set (?:it|them) as active"#), n > 0 {
            addDonFromDeck(ownerIsP0: ownerIsP0, count: n, active: true)
        }
        if let n = firstIntMatch(in: work, pattern: #"add up to (\d+) additional don!!? cards? and rest them"#), n > 0 {
            addDonFromDeck(ownerIsP0: ownerIsP0, count: n, active: false)
        }
        if let n = firstIntMatch(in: work, pattern: #"add up to (\d+) rested don!!? cards?"#), n > 0 {
            addDonFromDeck(ownerIsP0: ownerIsP0, count: n, active: false)
        }
        if let n = firstIntMatch(
            in: work,
            pattern: #"give\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?(?:\s+cards?)?\s+to\s+1 of your characters?"#
        ),
            n > 0
        {
            giveRestedDonFromCostToFirstCharacter(ownerIsP0: ownerIsP0, count: n)
        }
        if let n = firstIntMatch(
            in: work,
            pattern: #"give\s+up\s+to\s+(\d+)\s+rested\s+don(?:!!)?(?:\s+cards?)?\s+to\s+your\s+leader"#
        ),
            n > 0
        {
            giveRestedDonToLeaderFromCostArea(ownerIsP0: ownerIsP0, count: n)
        }
        if let n = firstIntMatch(
            in: work,
            pattern: #"give your leader and 1 character up to (\d+) rested don(?:!!)?(?:\s+cards?)?\s+each"#
        ),
            n > 0
        {
            let c = min(5, n)
            giveRestedDonToLeaderFromCostArea(ownerIsP0: ownerIsP0, count: c)
            giveRestedDonFromCostToFirstCharacter(ownerIsP0: ownerIsP0, count: c)
        }
        if let n = firstIntMatch(in: work, pattern: #"set up to (\d+) of your don!!? cards? as active"#), n > 0 {
            let cap = min(10, n)
            if work.contains("at the end of this turn") {
                if ownerIsP0 {
                    scheduledEndOfThisTurnDonRefreshP0.append(cap)
                } else {
                    scheduledEndOfThisTurnDonRefreshP1.append(cap)
                }
            } else {
                refreshDon(ownerIsP0: ownerIsP0, count: cap)
            }
        }

        if let n = firstIntMatch(in: work, pattern: #"add up to (\d+) cards? from the top of your deck to the top of your life cards?"#), n > 0 {
            gainLifeFromDeck(ownerIsP0: ownerIsP0, count: n)
        }
        if regexContains(work, #"add 1 card from the top of your deck to the top of your life cards?"#) {
            gainLifeFromDeck(ownerIsP0: ownerIsP0, count: 1)
        }

        if let threshold = firstIntMatch(in: work, pattern: #"k\.?o\.?\s+up to 1 of your opponent's characters? with a cost of (\d+) or less"#) {
            koOpponentCharacter(ownerIsP0: ownerIsP0, maxCost: threshold)
        }
        if let threshold = firstIntMatch(in: work, pattern: #"k\.?o\.?\s+up to 1 of your opponent's characters? with (\d+)\s*power or less"#) {
            koOpponentCharacterByPower(ownerIsP0: ownerIsP0, maxPower: threshold)
        }
        if work.contains("k.o. up to 1 of your opponent's") && work.contains("0 power or less") {
            koOpponentCharacterByPower(ownerIsP0: ownerIsP0, maxPower: 0)
        }
        if let pair = firstTwoIntMatch(
            in: work,
            pattern: #"rest up to (\d+) of your opponent's characters? with a cost of (\d+) or less"#
        ) {
            restOpponentCharacters(ownerIsP0: ownerIsP0, count: min(5, pair.0), maxCost: pair.1)
        } else if let threshold = firstIntMatch(
            in: work,
            pattern: #"rest up to 1 of your opponent's characters? with a cost of (\d+) or less"#
        ) {
            restOpponentCharacter(ownerIsP0: ownerIsP0, maxCost: threshold)
        } else if regexContains(work, #"rest up to 1 of your opponent's characters?"#) {
            restOpponentCharacter(ownerIsP0: ownerIsP0, maxCost: 99)
        }
        if let boost = firstIntMatch(in: work, pattern: #"give up to 1 of your leader or character cards?\s*\+(\d+)\s*power during this turn"#) {
            addPowerBoost(ownerIsP0: ownerIsP0, amount: boost)
        } else if let boost = firstIntMatch(in: work, pattern: #"your leader gets \+(\d+)\s*power during this turn"#) {
            addPowerBoostLeaderOnly(ownerIsP0: ownerIsP0, amount: boost)
        }

        if let amt = firstIntMatch(
            in: work,
            pattern: #"give up to 1 of your opponent's characters? -(\d+)\s*power during this turn"#
        ) {
            applyOppCharPowerDebuffFromEffect(ownerIsP0: ownerIsP0, amount: amt)
        }
        if let lim = firstIntMatch(
            in: work,
            pattern: #"trash up to 1 of your opponent's characters? with (\d+)\s*power or less"#
        ) {
            koOpponentCharacterByPower(ownerIsP0: ownerIsP0, maxPower: lim)
        }

        if let n = firstIntMatch(
            in: work,
            pattern: #"return up to 1 of your opponent's characters? with a cost of (\d+)\s+or\s+less to the owner's hand"#
        ) {
            returnOpponentCharacterToOwnerHand(ownerIsP0: ownerIsP0, maxCost: n)
        } else if let n = firstIntMatch(
            in: work,
            pattern: #"return up to 1 character with a cost of (\d+)\s+or\s+less to the owner's hand"#
        ) {
            returnOpponentCharacterToOwnerHand(ownerIsP0: ownerIsP0, maxCost: n)
        } else if let n = firstIntMatch(
            in: work,
            pattern: #"return up to 1 character with (\d+)\s+base power to the owner's hand"#
        ) {
            returnOpponentCharacterToOwnerHandByPower(ownerIsP0: ownerIsP0, maxPower: n)
        } else if work.contains("return 1 of their characters to the owner's hand")
            || work.contains("return 1 of your opponent's characters to the owner's hand")
        {
            returnOpponentCharacterToOwnerHand(ownerIsP0: ownerIsP0, maxCost: 99)
        }

        if let n = firstIntMatch(
            in: work,
            pattern: #"place up to (\d+) of your opponent's characters?.{0,100}at the bottom of the owner's deck"#
        ),
            n > 0
        {
            placeOpponentCharactersBottomOwnerDeck(ownerIsP0: ownerIsP0, count: min(2, n), maxCost: nil, maxPower: nil)
        }
        if let p = firstIntMatch(
            in: work,
            pattern: #"place up to \d+ of your opponent's characters? with (\d+)\s*power or less at the bottom of the owner's deck"#
        ) {
            placeOpponentCharactersBottomOwnerDeck(ownerIsP0: ownerIsP0, count: 1, maxCost: nil, maxPower: p)
        }
        if let c = firstIntMatch(
            in: work,
            pattern: #"place up to \d+ of your opponent's characters? with a cost of (\d+)\s+or\s+less at the bottom of the owner's deck"#
        ) {
            placeOpponentCharactersBottomOwnerDeck(ownerIsP0: ownerIsP0, count: 1, maxCost: c, maxPower: nil)
        }

        if let n = firstIntMatch(
            in: work,
            pattern: #"play up to (\d+) .* from your trash"#
        ),
            n > 0
        {
            let costMax = firstIntMatch(in: work, pattern: #"cost of (\d+)\s+or\s+less"#)
            let powerMax = firstIntMatch(in: work, pattern: #"(\d+)\s*power or less"#)
            playFromTrash(ownerIsP0: ownerIsP0, count: min(1, n), maxCost: costMax, maxPower: powerMax)
        }

        if work.contains("choose one:") {
            // Choix déterministe: priorise interaction board, sinon pioche.
            if regexContains(work, #"return up to 1 .*to the owner's hand"#) {
                if let n = firstIntMatch(in: work, pattern: #"cost of (\d+)\s+or\s+less to the owner's hand"#) {
                    returnOpponentCharacterToOwnerHand(ownerIsP0: ownerIsP0, maxCost: n)
                } else {
                    returnOpponentCharacterToOwnerHand(ownerIsP0: ownerIsP0, maxCost: 99)
                }
            } else if regexContains(work, #"k\.?o\.?\s+up to 1"#) {
                if let n = firstIntMatch(in: work, pattern: #"cost of (\d+)\s+or\s+less"#) {
                    koOpponentCharacter(ownerIsP0: ownerIsP0, maxCost: n)
                }
            } else if let n = firstIntMatch(in: work, pattern: #"draw\s+(\d+)\s+cards?"#) {
                drawCards(ownerIsP0: ownerIsP0, count: n)
            }
        }
    }

    private func applyOppCharPowerDebuffFromEffect(ownerIsP0: Bool, amount: Int) {
        guard amount > 0 else { return }
        if ownerIsP0, let i = p1.board.indices.first {
            p1OppDebuffedCharId = p1.board[i].cardId
            p1OppDebuffAmount = max(p1OppDebuffAmount, amount)
        } else if !ownerIsP0, let i = p0.board.indices.first {
            p0OppDebuffedCharId = p0.board[i].cardId
            p0OppDebuffAmount = max(p0OppDebuffAmount, amount)
        }
    }

    private func firstIntMatch(in text: String, pattern: String) -> Int? {
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: text, options: [], range: range), m.numberOfRanges >= 2 else { return nil }
        return Int(ns.substring(with: m.range(at: 1)))
    }

    private func firstTwoIntMatch(in text: String, pattern: String) -> (Int, Int)? {
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return nil }
        let ns = text as NSString
        let r = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: text, options: [], range: r), m.numberOfRanges >= 3,
              let a = Int(ns.substring(with: m.range(at: 1))), let b = Int(ns.substring(with: m.range(at: 2)))
        else { return nil }
        return (a, b)
    }

    private func regexContains(_ text: String, _ pattern: String) -> Bool {
        guard let re = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else { return false }
        let ns = text as NSString
        return re.firstMatch(in: text, options: [], range: NSRange(location: 0, length: ns.length)) != nil
    }

    // MARK: - Recherche / look deck (évolué : pas de parité stricte Python)

    private enum DeckTextRegex {
        static let searchWithShuffle: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"search\s+your\s+deck\s+for\s+up\s+to\s+(\d+)[\s\S]{0,450}?shuffle\s+your\s+deck"#,
                options: [.caseInsensitive, .dotMatchesLineSeparators]
            )
        }()
        static let searchOnly: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"\bsearch\s+your\s+deck\s+for\s+up\s+to\s+(\d+)\b"#,
                options: .caseInsensitive
            )
        }()
        static let lookPlayThenBottom: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"look\s+at\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck[\s\S]*?play\s+up\s+to[\s\S]*?place\s+the\s+rest\s+at\s+the\s+bottom\s+of\s+your\s+deck(?:\s+in\s+any\s+order)?"#,
                options: [.caseInsensitive, .dotMatchesLineSeparators]
            )
        }()
        static let lookAddThenBottom: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"look\s+at\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck[\s\S]*?add\s+(?:it|the\s+revealed\s+card|up\s+to\s+\d+\s+cards?)\s+to\s+your\s+hand[\s\S]*?place\s+the\s+rest\s+at\s+the\s+bottom\s+of\s+your\s+deck(?:\s+in\s+any\s+order)?"#,
                options: [.caseInsensitive, .dotMatchesLineSeparators]
            )
        }()
    }

    /// Retire la première occurrence du motif (évite de recroiser search/look dans les effets suivants).
    private func stripFirstMatch(in work: inout String, regex: NSRegularExpression) {
        let ns = work as NSString
        let fullR = NSRange(location: 0, length: ns.length)
        guard let m = regex.firstMatch(in: work, options: [], range: fullR) else { return }
        let before = ns.substring(to: m.range.location)
        let after = ns.substring(from: m.range.location + m.range.length)
        work = before + after
    }

    /// Une passe : search long → search court → look play → look add. Retourne vrai si un effet a été appliqué.
    private func applyOneDeckWindowPass(work: inout String, ownerIsP0: Bool) -> Bool {
        let ns = work as NSString
        let fullR = NSRange(location: 0, length: ns.length)
        if let m = DeckTextRegex.searchWithShuffle.firstMatch(in: work, options: [], range: fullR), m.numberOfRanges >= 2 {
            let n = Int(ns.substring(with: m.range(at: 1))) ?? 0
            if n > 0 {
                let win = ns.substring(with: m.range)
                searchDeckAddHand(ownerIsP0: ownerIsP0, maxTake: n, window: win)
            }
            stripFirstMatch(in: &work, regex: DeckTextRegex.searchWithShuffle)
            return true
        }
        if let m = DeckTextRegex.searchOnly.firstMatch(in: work, options: [], range: fullR), m.numberOfRanges >= 2 {
            let n = Int(ns.substring(with: m.range(at: 1))) ?? 0
            if n > 0 {
                let w = ns.substring(with: m.range) + " shuffle your deck"
                searchDeckAddHand(ownerIsP0: ownerIsP0, maxTake: n, window: w)
            }
            stripFirstMatch(in: &work, regex: DeckTextRegex.searchOnly)
            return true
        }
        for re in [DeckTextRegex.lookPlayThenBottom, DeckTextRegex.lookAddThenBottom] {
            let nlen = (work as NSString).length
            let r2 = NSRange(location: 0, length: nlen)
            guard let m2 = re.firstMatch(in: work, options: [], range: r2), m2.numberOfRanges >= 2,
                  let n = Int((work as NSString).substring(with: m2.range(at: 1))), n > 0
            else { continue }
            let win = (work as NSString).substring(with: m2.range)
            applyLookAtTop(ownerIsP0: ownerIsP0, peek: n, window: win)
            stripFirstMatch(in: &work, regex: re)
            return true
        }
        return false
    }

    /// Jusqu’à 4 effets deck consécutifs (ex. deux search sur la même clause), texte consommé entre chaque.
    private func applyDeckWindowEffectsLoop(work: inout String, ownerIsP0: Bool) {
        for _ in 0..<4 {
            if !applyOneDeckWindowPass(work: &work, ownerIsP0: ownerIsP0) { break }
        }
    }

    // MARK: - Composites main / deck (2-8-3 approximé : retrait de sous-chaîne pour chaque règle prise en priorité)

    private enum CompositeTextRegex {
        static let drawTrashAll: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+(\d+)\s+cards?\s+and\s+trash\s+(\d+)\s+cards?\s+from\s+your\s+hand"#,
                options: .caseInsensitive
            )
        }()
        static let drawTrashOne: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+(\d+)\s+cards?\s+and\s+trash\s+(?:a|1|one)\s+card\s+from\s+your\s+hand"#,
                options: .caseInsensitive
            )
        }()
        static let drawPlaceBottom: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+(\d+)\s+cards?\s+and\s+place\s+(\d+)\s+cards?\s+from\s+your\s+hand\s+at\s+the\s+bottom"#,
                options: .caseInsensitive
            )
        }()
        static let placeHandBottom: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"\bplace\s+(\d+)\s+cards?\s+from\s+your\s+hand\s+at\s+the\s+bottom"#,
                options: .caseInsensitive
            )
        }()
        static let trashDeckTop: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"trash\s+(\d+)\s+cards?\s+from\s+the\s+top\s+of\s+your\s+deck"#,
                options: .caseInsensitive
            )
        }()
        static let drawThenDiscard: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(\d+)\s+cards?"#,
                options: .caseInsensitive
            )
        }()
        static let drawThenDiscardCard: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(\d+)\s+card\b"#,
                options: .caseInsensitive
            )
        }()
        static let drawADiscardA: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+a\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+(?:a|1|one)\s+card"#,
                options: .caseInsensitive
            )
        }()
        static let draw1Discard1: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"draw\s+1\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*discard\s+1\s+card"#,
                options: .caseInsensitive
            )
        }()
        static let discardAThenDrawA: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"discard\s+(?:a|1|one)\s+card\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(?:a|1|one)\s+card"#,
                options: .caseInsensitive
            )
        }()
        static let discardThenDraw: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"discard\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(\d+)\s+cards?"#,
                options: .caseInsensitive
            )
        }()
        static let discardThenDrawCard: NSRegularExpression = {
            try! NSRegularExpression(
                pattern: #"discard\s+(\d+)\s+cards?\s*(?:,|;|\.|\s+then\s+|\s+and\s+)\s*draw\s+(\d+)\s+card\b"#,
                options: .caseInsensitive
            )
        }()
    }

    private func placeRandomHandCardsOnDeckBottom(ownerIsP0: Bool, n: Int) {
        for _ in 0..<max(0, n) {
            if ownerIsP0 {
                guard !p0.hand.isEmpty else { break }
                let i = Int(rng.nextUInt64() % UInt64(p0.hand.count))
                p0.deck.append(p0.hand.remove(at: i))
            } else {
                guard !p1.hand.isEmpty else { break }
                let i = Int(rng.nextUInt64() % UInt64(p1.hand.count))
                p1.deck.append(p1.hand.remove(at: i))
            }
        }
    }

    private func trashFromTopOfDeck(ownerIsP0: Bool, n: Int) {
        for _ in 0..<max(0, n) {
            if ownerIsP0, !p0.deck.isEmpty { p0.trash.append(p0.deck.removeFirst()) }
            else if !ownerIsP0, !p1.deck.isEmpty { p1.trash.append(p1.deck.removeFirst()) }
        }
    }

    /// Passe en priorité d’inscription des motifs (comme l’enchaînement de ``_RULES`` en Python).
    private func applyOneCompositeHandDeckPass(work: inout String, ownerIsP0: Bool) -> Bool {
        let ns0 = work as NSString
        let r0 = NSRange(location: 0, length: ns0.length)
        if let m = CompositeTextRegex.drawTrashAll.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let d = Int(ns0.substring(with: m.range(at: 1))) ?? 0
            let t = Int(ns0.substring(with: m.range(at: 2))) ?? 0
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            if t > 0 { discardSelf(ownerIsP0: ownerIsP0, count: t) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawTrashAll)
            return true
        }
        if let m = CompositeTextRegex.drawTrashOne.firstMatch(in: work, options: [], range: r0) {
            let d = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawTrashOne)
            return true
        }
        if let m = CompositeTextRegex.drawPlaceBottom.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let d = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            let b = Int((work as NSString).substring(with: m.range(at: 2))) ?? 0
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            if b > 0 { placeRandomHandCardsOnDeckBottom(ownerIsP0: ownerIsP0, n: b) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawPlaceBottom)
            return true
        }
        if let m = CompositeTextRegex.placeHandBottom.firstMatch(in: work, options: [], range: r0) {
            let p = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            if p > 0 { placeRandomHandCardsOnDeckBottom(ownerIsP0: ownerIsP0, n: p) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.placeHandBottom)
            return true
        }
        if let m = CompositeTextRegex.trashDeckTop.firstMatch(in: work, options: [], range: r0) {
            let t = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            if t > 0 { trashFromTopOfDeck(ownerIsP0: ownerIsP0, n: t) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.trashDeckTop)
            return true
        }
        if let m = CompositeTextRegex.drawThenDiscard.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let d = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            let x = Int((work as NSString).substring(with: m.range(at: 2))) ?? 0
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            if x > 0 { discardSelf(ownerIsP0: ownerIsP0, count: x) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawThenDiscard)
            return true
        }
        if let m = CompositeTextRegex.drawThenDiscardCard.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let d = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            let x = Int((work as NSString).substring(with: m.range(at: 2))) ?? 0
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            if x > 0 { discardSelf(ownerIsP0: ownerIsP0, count: x) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawThenDiscardCard)
            return true
        }
        if CompositeTextRegex.drawADiscardA.firstMatch(in: work, options: [], range: r0) != nil {
            drawCards(ownerIsP0: ownerIsP0, count: 1)
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
            stripFirstMatch(in: &work, regex: CompositeTextRegex.drawADiscardA)
            return true
        }
        if CompositeTextRegex.draw1Discard1.firstMatch(in: work, options: [], range: r0) != nil {
            drawCards(ownerIsP0: ownerIsP0, count: 1)
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
            stripFirstMatch(in: &work, regex: CompositeTextRegex.draw1Discard1)
            return true
        }
        if CompositeTextRegex.discardAThenDrawA.firstMatch(in: work, options: [], range: r0) != nil {
            discardSelf(ownerIsP0: ownerIsP0, count: 1)
            drawCards(ownerIsP0: ownerIsP0, count: 1)
            stripFirstMatch(in: &work, regex: CompositeTextRegex.discardAThenDrawA)
            return true
        }
        if let m = CompositeTextRegex.discardThenDraw.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let x = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            let d = Int((work as NSString).substring(with: m.range(at: 2))) ?? 0
            if x > 0 { discardSelf(ownerIsP0: ownerIsP0, count: x) }
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.discardThenDraw)
            return true
        }
        if let m = CompositeTextRegex.discardThenDrawCard.firstMatch(in: work, options: [], range: r0), m.numberOfRanges >= 3 {
            let x = Int((work as NSString).substring(with: m.range(at: 1))) ?? 0
            let d = Int((work as NSString).substring(with: m.range(at: 2))) ?? 0
            if x > 0 { discardSelf(ownerIsP0: ownerIsP0, count: x) }
            if d > 0 { drawCards(ownerIsP0: ownerIsP0, count: d) }
            stripFirstMatch(in: &work, regex: CompositeTextRegex.discardThenDrawCard)
            return true
        }
        return false
    }

    private func applyCompositeHandDeckEffectsLoop(work: inout String, ownerIsP0: Bool) {
        for _ in 0..<8 {
            if !applyOneCompositeHandDeckPass(work: &work, ownerIsP0: ownerIsP0) { break }
        }
    }

    private func searchDeckAddHand(ownerIsP0: Bool, maxTake: Int, window: String) {
        guard maxTake > 0 else { return }
        let wlow = window.lowercased()
        let costMax = parseCostMaxOrLess(wlow)
        let costMin = parseCostOrMore(wlow)
        let typeSub = parseTypeQuote(window) ?? parseBracketType(window)
        let exclude = parseOtherThanBracketIds(wlow)
        let wantChar = wlow.contains("character card") || wlow.contains("character cards")
        let wantEvent = wlow.range(of: #"\bevent\s+cards?\b"#, options: .regularExpression) != nil
            && !wantChar
        if ownerIsP0 {
            let (taken, rest) = pickFromSequence(
                ids: p0.deck,
                typeSub: typeSub,
                costMax: costMax,
                costMin: costMin,
                exclude: exclude,
                wantChar: wantChar,
                wantEvent: wantEvent,
                maxAdd: maxTake
            )
            p0.deck = rest
            for cid in taken { p0.hand.append(cid) }
            shuffle(&p0.deck)
        } else {
            let (taken, rest) = pickFromSequence(
                ids: p1.deck,
                typeSub: typeSub,
                costMax: costMax,
                costMin: costMin,
                exclude: exclude,
                wantChar: wantChar,
                wantEvent: wantEvent,
                maxAdd: maxTake
            )
            p1.deck = rest
            for cid in taken { p1.hand.append(cid) }
            shuffle(&p1.deck)
        }
    }

    private func applyLookAtTop(ownerIsP0: Bool, peek: Int, window: String) {
        let wlow = window.lowercased()
        var maxAdd = 1
        if let n = firstIntMatch(in: wlow, pattern: #"reveal\s+up\s+to\s+(\d+)"#) {
            maxAdd = min(7, n)
        }
        let costMax = parseCostMaxOrLess(wlow)
        let costMin = parseCostOrMore(wlow)
        let typeSub = parseTypeQuote(window) ?? parseBracketType(window)
        let exclude = parseOtherThanBracketIds(wlow)
        let wantChar = wlow.contains("character card") || wlow.contains("character cards")
        let wantEvent = wlow.range(of: #"\bevent\s+cards?\b"#, options: .regularExpression) != nil
            && !wantChar
        if ownerIsP0 {
            let npeek = min(max(0, peek), p0.deck.count)
            var top: [String] = []
            for _ in 0..<npeek where !p0.deck.isEmpty { top.append(p0.deck.removeFirst()) }
            let (taken, rrest) = pickFromSequence(
                ids: top,
                typeSub: typeSub,
                costMax: costMax,
                costMin: costMin,
                exclude: exclude,
                wantChar: wantChar,
                wantEvent: wantEvent,
                maxAdd: maxAdd
            )
            for cid in taken { p0.hand.append(cid) }
            p0.deck.insert(contentsOf: rrest, at: 0)
            if wlow.contains("shuffle your deck") { shuffle(&p0.deck) }
        } else {
            let npeek = min(max(0, peek), p1.deck.count)
            var top: [String] = []
            for _ in 0..<npeek where !p1.deck.isEmpty { top.append(p1.deck.removeFirst()) }
            let (taken, rrest) = pickFromSequence(
                ids: top,
                typeSub: typeSub,
                costMax: costMax,
                costMin: costMin,
                exclude: exclude,
                wantChar: wantChar,
                wantEvent: wantEvent,
                maxAdd: maxAdd
            )
            for cid in taken { p1.hand.append(cid) }
            p1.deck.insert(contentsOf: rrest, at: 0)
            if wlow.contains("shuffle your deck") { shuffle(&p1.deck) }
        }
    }

    private func typeBlobForCard(_ c: SwiftCard) -> String {
        "\(c.cardText) \(c.name) \(c.cardType)".lowercased()
    }

    private func typeMatchesSub(_ c: SwiftCard, fragment: String) -> Bool {
        let frag = fragment.trimmingCharacters(in: .whitespaces).replacingOccurrences(
            of: "\"", with: ""
        ).lowercased()
        if frag.isEmpty { return true }
        let b = typeBlobForCard(c)
        if b.contains(frag) { return true }
        let c2 = frag.replacingOccurrences(of: " ", with: "")
        if b.replacingOccurrences(of: " ", with: "").contains(c2) { return true }
        let t = c.cardText
        if let re = try? NSRegularExpression(
            pattern: #"(?:"|"){1,2}([^"']{2,100})(?:"|")\s*type"#,
            options: .caseInsensitive
        ) {
            let ns = t as NSString
            let range = NSRange(location: 0, length: ns.length)
            for m in re.matches(in: t, options: [], range: range) {
                if m.numberOfRanges >= 2 {
                    let inner = ns.substring(with: m.range(at: 1))
                        .lowercased()
                        .replacingOccurrences(of: " ", with: "")
                    if inner.contains(frag) || inner.contains(frag.replacingOccurrences(of: " ", with: "")) {
                        return true
                    }
                }
            }
        }
        return false
    }

    /// Comme ``re.sub(r"[^A-Z0-9]", "", ...)`` en Python (``_excludes_card``).
    private func asciiAZ09Only(_ s: String) -> String {
        s.uppercased().filter { ($0 >= "A" && $0 <= "Z") || ($0 >= "0" && $0 <= "9") }
    }

    private func excludesCard(cardId: String, cd: SwiftCard?, token: String) -> Bool {
        let cux = cardId.trimmingCharacters(in: .whitespaces).uppercased()
        let t = token.trimmingCharacters(in: .whitespaces).uppercased()
        if t.isEmpty { return false }
        if cux == t { return true }
        guard let cd, !t.isEmpty else { return false }
        let e = asciiAZ09Only(t)
        if e.isEmpty { return false }
        let nm = asciiAZ09Only(cd.name)
        if nm.hasPrefix(e) { return true }
        if e.count >= 4, nm.contains(e) { return true }
        return false
    }

    private func parseOtherThanBracketIds(_ wl: String) -> [String] {
        guard let re = try? NSRegularExpression(
            pattern: #"other\s+than\s+\[([^\]]+)\]"#,
            options: .caseInsensitive
        ) else { return [] }
        var out: [String] = []
        re.enumerateMatches(in: wl, options: [], range: NSRange(location: 0, length: (wl as NSString).length)) { m, _, _ in
            if let m, m.numberOfRanges >= 2 {
                let s = (wl as NSString).substring(with: m.range(at: 1))
                let x = s.trimmingCharacters(in: .whitespaces)
                if !x.isEmpty { out.append(x) }
            }
        }
        return out
    }

    private func parseTypeQuote(_ window: String) -> String? {
        if let s = firstStringMatch(in: window, pattern: "\"+([^\"]{2,120})\"+\\s*type") { return s }
        // Export tcgcsv : guillemets doublés ``""Name"" type``
        if let s = firstStringMatch(in: window, pattern: "\"{2,}([^\"]{2,120})\"{2,}\\s*type") { return s }
        return nil
    }

    private func firstStringMatch(in text: String, pattern: String) -> String? {
        guard let re = try? NSRegularExpression(
            pattern: pattern,
            options: .caseInsensitive
        ) else { return nil }
        let ns = text as NSString
        let range = NSRange(location: 0, length: ns.length)
        guard let m = re.firstMatch(in: text, options: [], range: range), m.numberOfRanges >= 2
        else { return nil }
        return ns.substring(with: m.range(at: 1))
    }

    private func parseBracketType(_ window: String) -> String? {
        if let s = firstStringMatch(
            in: window,
            pattern: #"reveal\s+up\s+to\s+\d+\s+\[([^\]]{1,80})\](?:\s*and\s*)?add"#
        ) { return s }
        if let s = firstStringMatch(in: window, pattern: #"\[([^\]]{1,80})\]\s*type"#) { return s }
        return nil
    }

    private func parseCostMaxOrLess(_ wl: String) -> Int? {
        for pat in [
            #"cost\s+of\s+(\d+)\s+or\s+less"#,
            #"with\s+a\s+cost\s+of\s+(\d+)\s+or\s+less"#,
            #"cost\s+(\d+)\s+or\s+less"#,
        ] {
            if let n = firstIntMatch(in: wl, pattern: pat) { return n }
        }
        return nil
    }

    private func parseCostOrMore(_ wl: String) -> Int? {
        for pat in [
            #"cost\s+of\s+(\d+)\s+or\s+more"#,
            #"with\s+a\s+cost\s+of\s+(\d+)\s+or\s+more"#,
        ] {
            if let n = firstIntMatch(in: wl, pattern: pat) { return n }
        }
        return nil
    }

    private func pickFromSequence(
        ids: [String],
        typeSub: String?,
        costMax: Int?,
        costMin: Int?,
        exclude: [String],
        wantChar: Bool,
        wantEvent: Bool,
        maxAdd: Int
    ) -> ([String], [String]) {
        var taken: [String] = []
        var rest: [String] = []
        let hasSub = (typeSub != nil) && !((typeSub ?? "").isEmpty)
        for cid in ids {
            if taken.count >= maxAdd { rest.append(cid); continue }
            guard let cd = cards[cid] else { rest.append(cid); continue }
            func excluded() -> Bool {
                for t in exclude where excludesCard(cardId: cid, cd: cd, token: t) { return true }
                return false
            }
            func ok(relaxType: Bool) -> Bool {
                if excluded() { return false }
                if wantChar, !cd.loweredType.contains("character") { return false }
                if wantEvent, !cd.loweredType.contains("event") { return false }
                if let cmax = costMax, cd.cost > cmax { return false }
                if let cmin = costMin, cd.cost < cmin { return false }
                if hasSub, !relaxType, let sub = typeSub, !typeMatchesSub(cd, fragment: sub) { return false }
                return true
            }
            if ok(relaxType: false) { taken.append(cid); continue }
            if hasSub, ok(relaxType: true) { taken.append(cid); continue }
            rest.append(cid)
        }
        return (taken, rest)
    }

    private func applyLifeTriggerIfAny(cardId: String, defenderIsP0: Bool) {
        guard let card = cards[cardId] else { return }
        applyTimingTextEffects(for: card, ownerIsP0: defenderIsP0, timing: "trigger")
    }

    private func flushDeferredLifeTriggers() {
        guard !deferredLifeTriggers.isEmpty else { return }
        let pending = deferredLifeTriggers
        deferredLifeTriggers.removeAll(keepingCapacity: true)
        for item in pending {
            applyLifeTriggerIfAny(cardId: item.cardId, defenderIsP0: item.defenderIsP0)
        }
    }

    private func preprocessConditionalText(_ text: String, ownerIsP0: Bool) -> String? {
        var work = text
        let turnNum = ownerIsP0 ? ((turnsStarted + 1) / 2) : (turnsStarted / 2)
        if work.contains("if it is your second turn or later") || work.contains("if it's your second turn or later") {
            if turnNum < 2 { return nil }
            work = work.replacingOccurrences(of: "if it is your second turn or later,", with: "")
            work = work.replacingOccurrences(of: "if it's your second turn or later,", with: "")
        }
        // Uniquement si le **segment** commence par la condition (évite d’invalider un bloc « A then if vous… » ex. OP05-019).
        if let n = firstIntMatch(in: work, pattern: #"^\s*if you have (\d+) or less life cards?"#) {
            let life = ownerIsP0 ? p0.lifeCards.count : p1.lifeCards.count
            if life > n { return nil }
        }
        if let n = firstIntMatch(in: work, pattern: #"^\s*if you have (\d+) or more cards? in your hand"#) {
            let hand = ownerIsP0 ? p0.hand.count : p1.hand.count
            if hand < n { return nil }
        }
        if let n = firstIntMatch(in: work, pattern: #"^\s*if your opponent has (\d+) or less life cards?"#) {
            let lifeOpp = ownerIsP0 ? p1.lifeCards.count : p0.lifeCards.count
            if lifeOpp > n { return nil }
        }
        return work
    }

    private func drawCards(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            for _ in 0..<count { if let c = drawCard(from: &p0) { p0.hand.append(c) } }
        } else {
            for _ in 0..<count { if let c = drawCard(from: &p1) { p1.hand.append(c) } }
        }
    }

    private func discardSelf(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            for _ in 0..<count where !p0.hand.isEmpty { p0.trash.append(p0.hand.removeLast()) }
        } else {
            for _ in 0..<count where !p1.hand.isEmpty { p1.trash.append(p1.hand.removeLast()) }
        }
    }

    private func discardOpponent(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            for _ in 0..<count where !p1.hand.isEmpty { p1.trash.append(p1.hand.removeLast()) }
        } else {
            for _ in 0..<count where !p0.hand.isEmpty { p0.trash.append(p0.hand.removeLast()) }
        }
    }

    private func trashOpponentLife(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            for _ in 0..<count where !p1.lifeCards.isEmpty { p1.trash.append(p1.lifeCards.removeLast()) }
        } else {
            for _ in 0..<count where !p0.lifeCards.isEmpty { p0.trash.append(p0.lifeCards.removeLast()) }
        }
    }

    private func addDonFromDeck(ownerIsP0: Bool, count: Int, active: Bool) {
        guard count > 0 else { return }
        if ownerIsP0 {
            let room = max(0, cfg.maxDon - (p0.donActive + p0.donRested))
            let gain = min(count, p0.donDeck, room)
            p0.donDeck -= gain
            if active { p0.donActive += gain } else { p0.donRested += gain }
        } else {
            let room = max(0, cfg.maxDon - (p1.donActive + p1.donRested))
            let gain = min(count, p1.donDeck, room)
            p1.donDeck -= gain
            if active { p1.donActive += gain } else { p1.donRested += gain }
        }
    }

    /// CR : des DON!! **reposés** (zone de coût) → attachés au **premier** personnage (comme ``_give_rested_don`` Python — Enel OP15 donne après avoir ajouté 4 en reposé).
    private func giveRestedDonFromCostToFirstCharacter(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            guard !p0.board.isEmpty, p0.donRested > 0 else { return }
            let t = min(count, p0.donRested)
            for _ in 0..<t {
                guard p0.donRested > 0 else { break }
                p0.donRested -= 1
                p0.board[0].attachedDon += 1
            }
        } else {
            guard !p1.board.isEmpty, p1.donRested > 0 else { return }
            let t = min(count, p1.donRested)
            for _ in 0..<t {
                guard p1.donRested > 0 else { break }
                p1.donRested -= 1
                p1.board[0].attachedDon += 1
            }
        }
    }

    private func giveRestedDonToLeaderFromCostArea(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0, p0.donRested > 0 {
            let t = min(count, p0.donRested)
            p0.donRested -= t
            p0.leaderAttachedDon += t
        } else if !ownerIsP0, p1.donRested > 0 {
            let t = min(count, p1.donRested)
            p1.donRested -= t
            p1.leaderAttachedDon += t
        }
    }

    private func refreshDon(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            let n = min(count, p0.donRested)
            p0.donRested -= n
            p0.donActive += n
        } else {
            let n = min(count, p1.donRested)
            p1.donRested -= n
            p1.donActive += n
        }
    }

    private func gainLifeFromDeck(ownerIsP0: Bool, count: Int) {
        guard count > 0 else { return }
        if ownerIsP0 {
            for _ in 0..<count { if let c = drawCard(from: &p0) { p0.lifeCards.insert(c, at: 0) } }
        } else {
            for _ in 0..<count { if let c = drawCard(from: &p1) { p1.lifeCards.insert(c, at: 0) } }
        }
    }

    private func koOpponentCharacter(ownerIsP0: Bool, maxCost: Int) {
        guard maxCost >= 0 else { return }
        if ownerIsP0 {
            if let idx = p1.board.firstIndex(where: { (cards[$0.cardId]?.cost ?? 99) <= maxCost }) {
                let dead = p1.board.remove(at: idx)
                p1.trash.append(dead.cardId)
            }
        } else {
            if let idx = p0.board.firstIndex(where: { (cards[$0.cardId]?.cost ?? 99) <= maxCost }) {
                let dead = p0.board.remove(at: idx)
                p0.trash.append(dead.cardId)
            }
        }
    }

    private func koOpponentCharacterByPower(ownerIsP0: Bool, maxPower: Int) {
        guard maxPower >= 0 else { return }
        if ownerIsP0 {
            if let idx = p1.board.firstIndex(where: { ($0.power + $0.attachedDon * 1000) <= maxPower }) {
                let dead = p1.board.remove(at: idx)
                p1.trash.append(dead.cardId)
            }
        } else {
            if let idx = p0.board.firstIndex(where: { ($0.power + $0.attachedDon * 1000) <= maxPower }) {
                let dead = p0.board.remove(at: idx)
                p0.trash.append(dead.cardId)
            }
        }
    }

    private func restOpponentCharacter(ownerIsP0: Bool, maxCost: Int) {
        restOpponentCharacters(ownerIsP0: ownerIsP0, count: 1, maxCost: maxCost)
    }

    private func restOpponentCharacters(ownerIsP0: Bool, count: Int, maxCost: Int) {
        guard count > 0 else { return }
        var n = count
        while n > 0 {
            if ownerIsP0 {
                guard let idx = p1.board.firstIndex(where: { !$0.rested && (cards[$0.cardId]?.cost ?? 99) <= maxCost })
                else { return }
                p1.board[idx].rested = true
            } else {
                guard let idx = p0.board.firstIndex(where: { !$0.rested && (cards[$0.cardId]?.cost ?? 99) <= maxCost })
                else { return }
                p0.board[idx].rested = true
            }
            n -= 1
        }
    }

    private func returnOpponentCharacterToOwnerHand(ownerIsP0: Bool, maxCost: Int) {
        if ownerIsP0 {
            guard let idx = p1.board.firstIndex(where: { (cards[$0.cardId]?.cost ?? 99) <= maxCost }) else { return }
            let ch = p1.board.remove(at: idx)
            p1.hand.append(ch.cardId)
        } else {
            guard let idx = p0.board.firstIndex(where: { (cards[$0.cardId]?.cost ?? 99) <= maxCost }) else { return }
            let ch = p0.board.remove(at: idx)
            p0.hand.append(ch.cardId)
        }
    }

    private func returnOpponentCharacterToOwnerHandByPower(ownerIsP0: Bool, maxPower: Int) {
        if ownerIsP0 {
            guard let idx = p1.board.firstIndex(where: { ($0.power + $0.attachedDon * 1000) <= maxPower }) else { return }
            let ch = p1.board.remove(at: idx)
            p1.hand.append(ch.cardId)
        } else {
            guard let idx = p0.board.firstIndex(where: { ($0.power + $0.attachedDon * 1000) <= maxPower }) else { return }
            let ch = p0.board.remove(at: idx)
            p0.hand.append(ch.cardId)
        }
    }

    private func placeOpponentCharactersBottomOwnerDeck(
        ownerIsP0: Bool,
        count: Int,
        maxCost: Int?,
        maxPower: Int?
    ) {
        guard count > 0 else { return }
        var left = count
        while left > 0 {
            if ownerIsP0 {
                guard let idx = p1.board.firstIndex(where: {
                    let costOk = maxCost == nil || (cards[$0.cardId]?.cost ?? 99) <= maxCost!
                    let powOk = maxPower == nil || ($0.power + $0.attachedDon * 1000) <= maxPower!
                    return costOk && powOk
                }) else { return }
                let ch = p1.board.remove(at: idx)
                p1.deck.append(ch.cardId)
            } else {
                guard let idx = p0.board.firstIndex(where: {
                    let costOk = maxCost == nil || (cards[$0.cardId]?.cost ?? 99) <= maxCost!
                    let powOk = maxPower == nil || ($0.power + $0.attachedDon * 1000) <= maxPower!
                    return costOk && powOk
                }) else { return }
                let ch = p0.board.remove(at: idx)
                p0.deck.append(ch.cardId)
            }
            left -= 1
        }
    }

    private func playFromTrash(ownerIsP0: Bool, count: Int, maxCost: Int?, maxPower: Int?) {
        guard count > 0 else { return }
        if ownerIsP0 {
            var left = min(count, max(0, cfg.maxBoard - p0.board.count))
            while left > 0 {
                guard let idx = p0.trash.indices.first(where: { i in
                    let cid = p0.trash[i]
                    guard let c = cards[cid], c.loweredType.contains("character") else { return false }
                    let costOk = maxCost == nil || c.cost <= maxCost!
                    let pOk = maxPower == nil || c.power <= maxPower!
                    return costOk && pOk
                }) else { return }
                let cid = p0.trash.remove(at: idx)
                if let c = cards[cid] {
                    p0.board.append(NativeBoardChar(card: c, justPlayed: true))
                }
                left -= 1
            }
        } else {
            var left = min(count, max(0, cfg.maxBoard - p1.board.count))
            while left > 0 {
                guard let idx = p1.trash.indices.first(where: { i in
                    let cid = p1.trash[i]
                    guard let c = cards[cid], c.loweredType.contains("character") else { return false }
                    let costOk = maxCost == nil || c.cost <= maxCost!
                    let pOk = maxPower == nil || c.power <= maxPower!
                    return costOk && pOk
                }) else { return }
                let cid = p1.trash.remove(at: idx)
                if let c = cards[cid] {
                    p1.board.append(NativeBoardChar(card: c, justPlayed: true))
                }
                left -= 1
            }
        }
    }

    private func addPowerBoost(ownerIsP0: Bool, amount: Int) {
        guard amount > 0 else { return }
        if ownerIsP0 {
            p0.leaderPowerBonusTurn += amount
        } else {
            p1.leaderPowerBonusTurn += amount
        }
    }

    private func addPowerBoostLeaderOnly(ownerIsP0: Bool, amount: Int) {
        addPowerBoost(ownerIsP0: ownerIsP0, amount: amount)
    }

    private func meetsDonXRequirement(
        ownerIsP0: Bool,
        required: Int,
        sourceIsLeader: Bool?,
        sourceBoardIndex: Int?,
        sourceCardId: String
    ) -> Bool {
        guard required > 0 else { return true }
        if ownerIsP0 {
            if sourceIsLeader == true { return p0.leaderAttachedDon >= required }
            if let i = sourceBoardIndex, i >= 0, i < p0.board.count { return p0.board[i].attachedDon >= required }
            if let c = p0.board.first(where: { $0.cardId == sourceCardId }) { return c.attachedDon >= required }
            return false
        } else {
            if sourceIsLeader == true { return p1.leaderAttachedDon >= required }
            if let i = sourceBoardIndex, i >= 0, i < p1.board.count { return p1.board[i].attachedDon >= required }
            if let c = p1.board.first(where: { $0.cardId == sourceCardId }) { return c.attachedDon >= required }
            return false
        }
    }

    private func leaderDonDeckSize(from leader: SwiftCard?, fallback: Int) -> Int {
        guard let leader else { return fallback }
        if leader.donDeckSize > 0, leader.donDeckSize != 10 {
            return leader.donDeckSize
        }
        if let n = firstIntMatch(in: leader.loweredText, pattern: #"don!!\s+deck\s+consists\s+of\s+(\d+)\s+cards?"#), n > 0 {
            return max(1, min(20, n))
        }
        return fallback
    }

    private func triggerLeaderAutoDrawForActivePlayer() {
        if activePlayerIsP0 {
            guard !p0.leaderOncePerTurnDrawUsed else { return }
            guard let lid = p0.leaderId, let c = cards[lid] else { return }
            let t = c.loweredText
            guard t.contains("[your turn]"), t.contains("when a card is removed"), t.contains("life"), t.contains("draw 1") else { return }
            if let drawn = drawCard(from: &p0) { p0.hand.append(drawn) }
            p0.leaderOncePerTurnDrawUsed = true
        } else {
            guard !p1.leaderOncePerTurnDrawUsed else { return }
            guard let lid = p1.leaderId, let c = cards[lid] else { return }
            let t = c.loweredText
            guard t.contains("[your turn]"), t.contains("when a card is removed"), t.contains("life"), t.contains("draw 1") else { return }
            if let drawn = drawCard(from: &p1) { p1.hand.append(drawn) }
            p1.leaderOncePerTurnDrawUsed = true
        }
    }

    private func triggerLeaderOpponentTurnLifeZeroIfNeeded(defender: inout NativePlayerState, defenderIsP0: Bool) {
        if !defender.lifeCards.isEmpty { return }
        if defenderIsP0 == activePlayerIsP0 { return }
        guard !defender.leaderOpponentTurnLifeZeroUsed else { return }
        guard let lid = defender.leaderId, let c = cards[lid] else { return }
        let t = c.loweredText
        guard t.contains("[opponent's turn]"), t.contains("life cards"), t.contains("becomes 0") else { return }
        defender.leaderOpponentTurnLifeZeroUsed = true
        if let d = drawCard(from: &defender) { defender.lifeCards.append(d) }
        if !defender.hand.isEmpty { defender.trash.append(defender.hand.removeLast()) }
    }

    private func applyLeaderDefensiveBoostIfAny(defender: inout NativePlayerState) {
        guard !defender.leaderOncePerTurnDefenseUsed else { return }
        guard let lid = defender.leaderId, let c = cards[lid] else { return }
        let t = c.loweredText
        guard t.contains("[on your opponent's attack]"), t.contains("+2000 power") else { return }
        let donReq = t.contains("[don!! x1]") ? 1 : 0
        guard defender.leaderAttachedDon >= donReq else { return }
        guard !defender.hand.isEmpty else { return }
        defender.trash.append(defender.hand.removeLast())
        defender.leaderPowerBonusTurn += 2000
        defender.leaderOncePerTurnDefenseUsed = true
    }

    private func observation() -> [Float] {
        var obs = [Float]()
        func add(_ v: Float) { obs.append(v) }
        add(Float(p0.lifeCards.count) / Float(max(1, cfg.startingLife)))
        add(Float(p1.lifeCards.count) / Float(max(1, cfg.startingLife)))
        add(Float(p0.hand.count) / 12.0)
        add(Float(p1.hand.count) / 12.0)
        add(Float(p0.board.count) / Float(max(1, cfg.maxBoard)))
        add(Float(p1.board.count) / Float(max(1, cfg.maxBoard)))
        add(Float(p0.donActive) / Float(max(1, cfg.maxDon)))
        add(Float(p1.donActive) / Float(max(1, cfg.maxDon)))
        add(Float(turnsStarted) / 40.0)
        add(phase == .main ? 1 : 0)
        add(phase == .battle ? 1 : 0)
        add(p0.leaderRested ? 1 : 0)
        add(p1.leaderRested ? 1 : 0)
        add(Float(p0.leaderAttachedDon) / 10.0)
        add(Float(p1.leaderAttachedDon) / 10.0)
        add(done ? 1 : 0)

        for i in 0..<SimConstants.handSlots {
            if i < p0.hand.count, let c = cards[p0.hand[i]] {
                obs.append(contentsOf: c.embedding(dim: SimConstants.embDim))
            } else {
                obs.append(contentsOf: [Float](repeating: 0, count: SimConstants.embDim))
            }
        }

        let scalarsCap = max(0, cfg.obsDim - SimConstants.handSlots * SimConstants.embDim)
        if obs.count > scalarsCap {
            obs = Array(obs.prefix(scalarsCap))
        } else if obs.count < scalarsCap {
            obs.append(contentsOf: [Float](repeating: 0, count: scalarsCap - obs.count))
        }
        if obs.count < cfg.obsDim {
            obs.append(contentsOf: [Float](repeating: 0, count: cfg.obsDim - obs.count))
        } else if obs.count > cfg.obsDim {
            obs = Array(obs.prefix(cfg.obsDim))
        }
        return obs
    }

    public func exportReplayState() -> ReplayState {
        func mapPlayer(_ p: NativePlayerState) -> ReplayPlayer {
            let board = p.board.map {
                ReplayBoardCard(
                    id: $0.cardId,
                    rested: $0.rested,
                    power: $0.power,
                    powerEffective: $0.power + $0.attachedDon * 1000,
                    hasRush: $0.hasRush || $0.hasRushChar,
                    hasBlocker: $0.hasBlocker,
                )
            }
            return ReplayPlayer(
                deckRemaining: p.deck.count,
                life: p.lifeCards.count,
                lifeCards: p.lifeCards,
                hand: p.hand,
                stageCardId: p.stageCardId,
                board: board,
                trash: p.trash,
                leaderId: p.leaderId,
                leaderPower: p.leaderPower,
                leaderPowerEffective: p.leaderPower + p.leaderAttachedDon * 1000,
                leaderRested: p.leaderRested,
                leaderAttachedDon: p.leaderAttachedDon,
                donActive: p.donActive,
                donRested: p.donRested,
                donDeck: p.donDeck,
                officialDonDeck: cfg.maxDon,
            )
        }

        let winner: Int?
        if !done {
            winner = nil
        } else if p1.lifeCards.isEmpty {
            winner = 0
        } else if p0.lifeCards.isEmpty {
            winner = 1
        } else {
            winner = nil
        }

        return ReplayState(
            phase: String(describing: phase),
            done: done,
            p0: mapPlayer(p0),
            p1: mapPlayer(p1),
            winner: winner,
        )
    }

    public static func describeAction(_ action: Int) -> String {
        if action == SimConstants.mainEndAction { return "main_end" }
        if action == SimConstants.battlePassAction { return "battle_pass" }
        if (2...6).contains(action) { return "play_hand_slot_\(action - 2)" }
        if action >= SimConstants.battleAttackBase && action < SimConstants.mainAttachDonBase {
            let rel = action - SimConstants.battleAttackBase
            return "attack_a\(rel / SimConstants.nTargets)_t\(rel % SimConstants.nTargets)"
        }
        if action >= SimConstants.mainAttachDonBase && action < SimConstants.mainActivateMainBase {
            let rel = action - SimConstants.mainAttachDonBase
            return "attach_slot_\(rel / SimConstants.mainAttachDonMax)_don_\((rel % SimConstants.mainAttachDonMax) + 1)"
        }
        if action >= SimConstants.mainActivateMainBase && action < SimConstants.mainActivateMainLeader {
            return "activate_main_slot_\(action - SimConstants.mainActivateMainBase)"
        }
        if action == SimConstants.mainActivateMainLeader { return "activate_main_leader" }
        return "action_\(action)"
    }
}
