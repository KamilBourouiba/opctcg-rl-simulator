import Foundation

/// Constantes d’action alignées sur ``opctcg_text_sim/simulator.py``.
public enum SimConstants {
    public static let mainEndAction = 7
    public static let battlePassAction = 8
    public static let battleAttackBase = 9
    public static let nAttackers = 6
    public static let nTargets = 6
    public static let nBattleAttackCodes = nAttackers * nTargets
    public static let mainAttachDonBase = battleAttackBase + nBattleAttackCodes
    public static let mainAttachDonSlots = 6
    public static let mainAttachDonMax = 10
    public static let mainAttachDonActions = mainAttachDonSlots * mainAttachDonMax
    public static let mainActivateMainBase = mainAttachDonBase + mainAttachDonActions
    public static let mainActivateMainSlots = 5
    public static let mainActivateMainLeader = mainActivateMainBase + mainActivateMainSlots
    public static let actionSpaceSize = mainActivateMainLeader + 1

    public static let mulliganKeep = 0
    public static let mulliganTake = 1

    public static let blockerPass = 0
    public static let blockerSlotBase = 1
    public static let blockerNSlots = 6

    public static let handSlots = 7
    public static let embDim = 8
}

public enum SimPhase: Equatable {
    case mulligan
    case main
    case battle
    case blocker
}
