import Foundation
import OPTrainingCore

/// Lance une démo d’entraînement **Swift pur** (régression linéaire + SGD) avec règles JSON — sans Python.
///
/// Usage :
///   swift run optrain
///   swift run optrain --config /chemin/vers/training_rules.json
enum Exit: Int32 {
    case ok = 0
    case usage = 1
    case io = 2
}

@main
struct OpTrainMain {
    static func main() {
        let argv = Array(CommandLine.arguments.dropFirst())
        var configPath: String?

        var i = argv.startIndex
        while i < argv.endIndex {
            let a = argv[i]
            if a == "--config" || a == "-c" {
                guard i + 1 < argv.endIndex else {
                    fputs("optrain: valeur manquante après --config\n", stderr)
                    exit(Exit.usage.rawValue)
                }
                configPath = argv[i + 1]
                i = argv.index(i, offsetBy: 2)
                continue
            }
            if a == "--help" || a == "-h" {
                print(
                    """
                    optrain — démo CPU (Swift) + règles JSON

                      swift run optrain
                      swift run optrain --config training_rules.json

                    Sans --config : utilise les règles embarquées (Resources/default_training.json).
                    """
                )
                exit(Exit.ok.rawValue)
            }
            fputs("optrain: argument inconnu `\(a)`\n", stderr)
            exit(Exit.usage.rawValue)
        }

        let rules: TrainingRules
        do {
            if let p = configPath {
                let url = URL(fileURLWithPath: p, isDirectory: false)
                rules = try TrainingRulesLoader.load(from: url)
            } else {
                rules = try TrainingRulesLoader.loadBundledDefault()
            }
        } catch {
            fputs("optrain: lecture config : \(error)\n", stderr)
            exit(Exit.io.rawValue)
        }

        fputs(
            """
            [optrain] chemins (référence projet — pas chargés ici)
              rules_corpus_out → \(rules.paths.rulesCorpusOut)
              cards_csv       → \(rules.paths.cardsCsv)
              rules_pdf       → \(rules.paths.rulesPdf)

            """,
            stderr,
        )

        let loss = SwiftCPUDemoTrainer.run(rules: rules, logEvery: 400)
        print("{\"ok\":true,\"final_mse\":\(loss),\"schema\":\(rules.schemaVersion)}")
        exit(Exit.ok.rawValue)
    }
}
