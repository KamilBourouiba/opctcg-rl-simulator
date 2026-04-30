import Foundation

/// PRNG léger — sans dépendance système.
public struct SplitMix64 {
    private var state: UInt64

    public init(seed: UInt64) {
        self.state = seed &+ 0x9E3779B97F4A7C15
    }

    public mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    public mutating func nextDouble(in range: ClosedRange<Double>) -> Double {
        let u = Double(nextUInt64() % 10_000) / 9999.0
        return range.lowerBound + (range.upperBound - range.lowerBound) * u
    }

    public mutating func nextGaussian() -> Double {
        let u1 = max(1e-12, Double(nextUInt64() % 1_000_000) / 1_000_000.0)
        let u2 = Double(nextUInt64() % 1_000_000) / 1_000_000.0
        return sqrt(-2.0 * log(u1)) * cos(2.0 * Double.pi * u2)
    }
}
