// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OPCTCGPolicySwift",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "OPCTCGPolicy", targets: ["OPCTCGPolicy"]),
        .executable(name: "op-policy-infer", targets: ["op-policy-infer"]),
    ],
    targets: [
        .target(
            name: "OPCTCGPolicy",
            path: "Sources/OPCTCGPolicy"
        ),
        .executableTarget(
            name: "op-policy-infer",
            dependencies: ["OPCTCGPolicy"],
            path: "Sources/op-policy-infer"
        ),
    ]
)
