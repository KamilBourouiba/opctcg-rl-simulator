import Foundation

/// Contrat commun au pont Python et au simulateur Swift natif.
public protocol TextSimEnvironment: AnyObject {
    func reset(seed: Int?) throws -> (obs: [Float], mask: [Bool])
    func step(action: Int) throws -> (obs: [Float], mask: [Bool], reward: Float, done: Bool)
    func close() throws
}
