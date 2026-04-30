# ONE PIECE CARD GAME — Référence de règles (Comprehensive Rules)

> Synthèse structurée à partir du document **Comprehensive Rules v1.2.0** (janvier 2026), fichier `rule_comprehensive.pdf`.  
> En cas d’écart, le texte des cartes prime sur les règles générales (règle 1-3-1).

---

## 1. Vue d’ensemble

### 1.1 Joueurs
- Partie **à deux** en tête-à-tête (1-1-1).

### 1.2 Fin de partie — conditions de défaite (1-2-1-1)
1. **0 cartes Vie** et le Leader subit des dégâts (1-2-1-1-1).
2. **0 cartes dans le deck** (1-2-1-1-2).

La défaite est appliquée au prochain traitement de règles (1-2-2).  
**Concession** : possible à tout moment ; elle n’est pas annulable par un effet (1-2-3 à 1-2-4).

### 1.3 Principes fondamentaux
- Action impossible → ignorée (1-3-2).
- **Coût** : paiement pour jouer une carte (nombre en haut à gauche) (1-3-9-1).
- **Coût d’activation** : pour activer un effet (1-3-9-2 ; voir section 8 du PDF complet).

---

## 2. Catégories et informations carte

### 2.1 Catégories (2-2-2)
**Leader**, **Character**, **Event**, **Stage**, **DON!!**.

### 2.2 Couleurs (2-3)
Six couleurs : rouge, vert, bleu, violet, noir, jaune ; cartes multicolores comptent pour toutes leurs couleurs (2-3-4).

### 2.3 Puissance, coût, texte (2-6 à 2-8)
- **Puissance** : force en combat (Leaders et Characters).
- **Coût** : pour jouer depuis la main — sélectionner des **DON!! actifs** dans la zone de coût, les **reposer**, puis jouer la carte (2-7-2 à 2-7-4).
- **Texte** : effets résolus du haut vers le bas (2-8-3). Le backend **Swift** (`OPTCGSimulator`, `applyTimingTextEffects`) modélise cela en enchaînant : (1) effets deck (search / look), (2) **motifs composites** du type « draw + trash », « place au dessous du deck », « mill » — chaque morceau est retiré du texte traité — puis (3) effets simples restants (pioche, défausse, DON, etc.), proches de `on_play_resolver.py` sans être identique au moteur Python complet.

### 2.4 Vie (2-9)
Valeur sur le Leader. Au début : tirer autant de cartes du deck que la Vie indiquée et les placer **face cachée** dans la zone Vie (5-2-1-7, 2-9-2).  
Les dégâts au Leader déplacent des cartes Vie vers la main (traitement des dégâts, 4-6).

---

## 3. Zones de jeu (3-1-1)

Aires : **deck**, **deck DON!!**, **main**, **trash**, **Leader**, **Character**, **Stage**, **zone de coût**, **zone Vie**.

- **Deck** : zone secrète, pile face cachée (3-2).
- **Deck DON!!** : zone ouverte, 10 cartes DON!! au départ (3-3, 5-1-2).
- **Zone de coût** : les DON!! utilisés pour payer les coûts ; certains effets les **donnent** aux Leaders/Characters (+1000 par DON donné pendant votre tour) (3-9, 6-5-5).

---

## 4. Mise en place (5. Game Setup)

### 4.1 Composition (5-1-2)
- **1** Leader  
- **50** cartes dans le deck (Characters, Events, Stages) — couleurs compatibles avec le Leader ; **max 4** exemplaires d’un même numéro (5-1-2-3).  
- **10** cartes dans le deck DON!! (5-1-2).

### 4.2 Début de partie (5-2-1)
1. Présenter et mélanger les decks (5-2-1-1, 5-2-1-2).  
2. Placer le Leader face visible (5-2-1-3).  
3. Premier joueur : déterminer qui commence (5-2-1-4).  
4. **Main de départ** : piocher **5** cartes ; **mulligan** une fois chacun, en commençant par le premier joueur (5-2-1-6).  
5. Placer les cartes **Vie** depuis le dessus du deck (5-2-1-7).  
6. Le **premier joueur commence** son tour (5-2-1-8).

---

## 5. Déroulement d’un tour (6. Game Progression)

### 5.1 Ordre des phases (6-1-1)
Un tour = **Refresh** → **Draw** → **DON!!** → **Main** → **End**.

### 5.2 Refresh (6-2)
- Fin des effets « jusqu’au début de votre prochain tour » (6-2-1).  
- Effets « au début du tour » (6-2-2).  
- Rappeler les DON!! donnés aux cartes vers la zone de coût **reposés** (6-2-3).  
- **Remettre en actif** (vertical) Leader, Character, Stage, coût (6-2-4).

### 5.3 Draw (6-3)
- Piocher **1** carte — **sauf** le **premier tour du premier joueur** (6-3-1).

### 5.4 Phase DON!! (6-4)
- Placer **2** DON!! du deck DON!! dans la zone de coût **face visible (actifs)** — **1 seul** au **premier tour** du premier joueur (6-4-1).  
- Si le deck DON!! est vide, ne rien placer (6-4-3).

### 5.5 Main Phase (6-5)
Actions possibles (dans n’importe quel ordre, autant de fois que souhaité) :  
- **Jouer** une carte (Character / Stage) ou activer un Event **[Main]** (6-5-3).  
- **Activer** des effets **[Main]** / **[Activate: Main]** (6-5-4).  
- **Donner** des DON!! (+1000 par DON) (6-5-5).  
- **Combattre** (6-5-6) — **interdit au premier tour** pour les deux joueurs (6-5-6-1).  
Fin de Main → **End Phase** (6-5-2-1).

### 5.6 End Phase (6-6)
Effets de fin de tour, invalidation des effets « pendant ce tour », puis passage du tour à l’adversaire → **Refresh** (6-6-1-4).

---

## 6. Combats (7. Card Attacks and Battles)

- En **Main**, le joueur actif peut **attaquer** avec un Leader ou Character **actif** : cible = Leader adverse ou Character **reposé** de l’adversaire (7-1).  
- Déroulement type : **Attack** → réactions (Counter, Blocker, etc.) → **Combat** → **Damage** (7-1-1 à 7-1-4 dans le PDF).  
- Les détails (contre-attaque, triggers, mots-clés) sont dans les sections **7**, **8** et **10** du PDF.

---

## 7. Traitement des dégâts (4-6)

Dégâts au Leader : pour chaque point, déplacer **1 carte** du dessus de la **zone Vie** vers la **main** (4-6-2-1).  
Si une carte **[Trigger]** arrive en main ainsi, le joueur peut l’activer (4-6-3).

---

## 8. Cartographie : ce que fait `opctcg_text_sim` (simulateur texte)

Le moteur `SimplifiedOPSim` est une **abstraction d’entraînement**, pas un moteur complet OP :

| Élément officiel | Comportement dans le sim |
|------------------|---------------------------|
| Phases Refresh / Draw / DON!! / End | **Non modélisées** ; enchaînement **MAIN** (jouer depuis la main ou passer) puis **BATTLE** simplifié. |
| Zone DON!! (10 cartes, actif/reposé) | **`don_deck`** (10) + **`don`** actifs en zone coût ; +`don_per_turn` depuis le deck sauf **1er DON phase du 1er joueur = 1** (6-4-1). L’adversaire qui commence son 1er tour reçoit **2** DON!! depuis son deck. |
| Premier Draw (6-3-1) | Option **`skip_first_draw_first_player`** : le 1er joueur ne pioche pas à son tout premier Draw Phase. |
| Combat 1er tour (6-5-6-1) | **`battle_damage_allowed`** après deux démarrages de tour (P0 puis P1). |
| Vie (pile + dégâts) | **Entier `life`** ; baisse si attaque « réussit » selon règle simplifiée (`leader_block`). |
| Deck 50 + défaite deck out | **Taille deck** exposée dans les replays pour affichage ; la défaite par deck out peut être ajoutée plus tard. |
| Main max | **7** cartes (limite de pioche). |
| Terrain | **Jusqu’à `max_board`** Characters ; états **actif / reposé** pour le rendu. |

Les GIF / JSONL de replay utilisent ces états pour **visualiser** (mains des deux joueurs, piles deck, DON!! en pastilles, vie) en s’inspirant de la disposition des zones **officielles** décrites ci-dessus.

---

## 9. Fichiers du projet

| Fichier | Rôle |
|---------|------|
| `rule_comprehensive.pdf` | Source officielle (à placer où vous voulez ; `~/Downloads/` par défaut dans la config). |
| `scripts/extract_rules_pdf.py` | Extraction texte brute → `data/rules_corpus.txt`. |
| Ce fichier (`data/OPTCG_GameRules_Reference.md`) | Référence lisible + lien avec le sim. |
| `opctcg_text_sim/rules_simple.py` | En-têtes de log et hachage optionnel du corpus + référence. |

Pour régénérer le corpus texte depuis le PDF :

```bash
python scripts/extract_rules_pdf.py ~/Downloads/rule_comprehensive.pdf -o data/rules_corpus.txt
```

Liste détaillée des **mots-clés** (`[On Play]`, `[Counter]`, `[DON!! x2]`, effets à mots-clés [Rush], etc.) : **`data/OPTCG_Keywords_Reference.md`**.
