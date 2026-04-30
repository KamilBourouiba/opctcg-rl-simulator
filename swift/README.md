# Inférence policy — **Swift uniquement**

À l’exécution, **aucun runtime Python** : uniquement le compilateur Swift, Foundation et (optionnellement) Core ML sur macOS.

## Prérequis

- macOS 13+
- Xcode ou **Swift** en ligne de commande (`xcode-select --install`)

## Compiler

```bash
cd swift
swift build -c release
```

## Lancer l’inférence

Le binaire **`op-policy-infer`** attend un **dossier bundle** et un vecteur d’observation (floats séparés par des virgules).

### Backend « float » pur Swift

Contenu du dossier (export côté repo Python : `scripts/export_policy_swift_bundle.py`) :

- `manifest.json`
- `weights.bin`

```bash
swift run -c release op-policy-infer /chemin/vers/swift_export "0.1,0.2,0.3"
```

### Backend Core ML (ANE / GPU)

Contenu du dossier (export : `scripts/export_policy_coreml.py`) :

- `coreml_manifest.json`
- `PolicyPi.mlpackage`
- `Val.mlpackage` (nom peut varier selon le manifest)

```bash
swift run -c release op-policy-infer /chemin/vers/coreml_bundle "0.1,0.2,..." --mask "111110..."
```

Le masque est une chaîne de longueur `n_act` : `1` = action légale, `0` = illégale.

### Sortie

Une ligne JSON (`stdout`) avec notamment `backend` (`swift_float` ou `coreml`), `logits`, `value`, `argmax_masked`.

## Raccourci : script sans Python

```bash
cd swift
chmod +x infer.sh
./infer.sh /chemin/bundle "0,0,0"
```

## Préparer le bundle

La **conversion checkpoint → dossier** utilise encore les scripts Python du dépôt (`export_policy_swift_bundle.py`, `export_policy_coreml.py`). Une fois le dossier copié sur une machine sans Python, seule la chaîne Swift ci-dessus suffit.
