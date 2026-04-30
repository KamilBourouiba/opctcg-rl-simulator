import Foundation

public struct SwiftPolicyManifest: Codable {
    public let schema_version: Int
    public let neg_inf_mask: Float
    public let obs_dim: Int
    public let hidden_dim: Int
    public let n_act: Int
    public let n_res_blocks: Int
    public let dtype: String
    public let endian: String
    public let segments: [SwiftPolicySegment]
    public let weights_file: String
}

public struct SwiftPolicySegment: Codable {
    public let name: String
    public let byte_offset: Int
    public let byte_length: Int
    public let shape: [Int]
}
