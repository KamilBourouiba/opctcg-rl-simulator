// swift-tools-version: 5.12
import PackageDescription

/// Swift : règles JSON, démo CPU, pont JSON vers le simulateur Python (fidélité complète),
/// REINFORCE + baseline sur politique linéaire.
let package = Package(
    name: "OPSwiftTraining",
    platforms: [.macOS(.v13)],
    products: [
        .library(name: "OPTrainingCore", targets: ["OPTrainingCore"]),
        .library(name: "OPTCGBridge", targets: ["OPTCGBridge"]),
        .library(name: "OPTCGRL", targets: ["OPTCGRL"]),
        .library(name: "OPTCGSimulator", targets: ["OPTCGSimulator"]),
        .executable(name: "optrain", targets: ["optrain"]),
        .executable(name: "optcg_train", targets: ["optcg_train"]),
        .executable(name: "optcg_parity", targets: ["optcg_parity"]),
    ],
    targets: [
        .target(
            name: "OPTrainingCore",
            path: "Sources/OPTrainingCore",
            resources: [.process("Resources")]
        ),
        .target(
            name: "OPTCGBridge",
            dependencies: ["OPTrainingCore"],
            path: "Sources/OPTCGBridge"
        ),
        .target(
            name: "OPTCGRL",
            dependencies: ["OPTrainingCore"],
            path: "Sources/OPTCGRL"
        ),
        .target(
            name: "OPTCGSimulator",
            dependencies: ["OPTrainingCore"],
            path: "Sources/OPTCGSimulator"
        ),
        .executableTarget(
            name: "optrain",
            dependencies: ["OPTrainingCore"],
            path: "Sources/optrain"
        ),
        .executableTarget(
            name: "optcg_train",
            dependencies: ["OPTrainingCore", "OPTCGBridge", "OPTCGRL", "OPTCGSimulator"],
            path: "Sources/optcg_train"
        ),
        .executableTarget(
            name: "optcg_parity",
            dependencies: ["OPTrainingCore", "OPTCGBridge", "OPTCGSimulator"],
            path: "Sources/optcg_parity"
        ),
    ]
)
