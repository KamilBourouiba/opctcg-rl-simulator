import Foundation

/// Point d’entrée futur pour le port du fichier Python ``simulator.py`` (~2100 lignes).
///
/// Déjà disponibles dans ce module :
/// - ``SimConstants`` — même grille d’actions que Python (`ACTION_SPACE_SIZE` = 111).
/// - ``SwiftCard`` + ``SwiftCSV.loadCards`` — lecture CSV tcgcsv (sans pandas).
/// - ``SwiftDeckIO`` — parse decks `.txt` et extraction Leader (alignés sur ``deck_parser.py``).
///
/// La mécanique de tour complète (effets « engine », counters, blocker pause, self-play)
/// sera ajoutée ici progressivement ; pour l’instant l’entraînement fidèle passe par
/// ``scripts/swift_env_server.py`` + ``PythonEnvSession``.
public enum NativeSimulatorRoadmap {}
