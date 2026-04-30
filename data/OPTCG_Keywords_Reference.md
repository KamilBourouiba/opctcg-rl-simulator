# ONE PIECE CARD GAME — Mots-clés et effets à mots-clés

> Référence synthétique d’après **Comprehensive Rules v1.2.0** (sections **8** effets / **10** mots-clés).  
> Les cartes anglaises utilisent la syntaxe `[ … ]`, `{ … }` (types), `< … >` (attributs). Le projet les extrait automatiquement du texte d’effet (`card_keywords.py`).

---

## A. Effets à mots-clés (*Keyword effects*, section 10-1)

Ce sont des **règles spéciales** portées par la carte ; elles sont listées en 10-1 du document officiel.

| Marqueur | Rôle (résumé officiel) |
|----------|-------------------------|
| **[Rush]** | Le Character peut **attaquer le même tour** où il est joué (10-1-1). |
| **[Double Attack]** | Si des dégâts sont infligés à la **Vie du Leader** adverse par cette attaque, **2 dégâts** au lieu de 1 (10-1-2). |
| **[Banish]** | Dégâts à la Vie du Leader adverse : une carte Vie est **défaussée** au lieu d’aller en main ; **[Trigger]** ne s’active pas (10-1-3). |
| **[Blocker]** | Quand une autre de vos cartes est attaquée, vous pouvez l’activer en **reposant** cette carte à l’étape Block ; elle devient la cible (10-1-4). |
| **[Trigger]** | En subissant des dégâts avec une carte **[Trigger]** en Vie : vous pouvez la **révéler** et activer son **[Trigger]** au lieu de l’ajouter à la main (10-1-5). |
| **[Rush: Character]** | Le Character peut **attaquer les Characters** adverses le tour où il est joué (10-1-6). |
| **[Unblockable]** | L’adversaire **ne peut pas activer [Blocker]** contre cette attaque (10-1-7). |

---

## B. Mots-clés (*Keywords*, section 10-2)

Ils précisent **timing**, **condition**, **coût d’activation** ou **vocabulaire de règle**.

### B.1 Timings d’effets automatiques (*auto effects*, voir aussi 8-1-3-1-1)

| Marqueur | Signification |
|----------|----------------|
| **[On Play]** | L’effet se déclenche **quand la carte est jouée** (10-2-6). |
| **[When Attacking]** | Déclenché quand vous **déclarez une attaque** à l’Attack Step (10-2-5). |
| **[On Block]** | Déclenché à l’étape Block quand vous activez votre **[Blocker]** (10-2-15). |
| **[On Your Opponent’s Attack]** | Déclenché quand l’adversaire déclare une attaque ; après ses effets « when attacking » (10-2-16). |
| **[On K.O.]** | Quand le Character est **K.O.** sur le terrain ; la carte va au trash entre activation et résolution (10-2-17). |
| **[End of Your Turn]** | **End Phase** de votre tour (10-2-7). |
| **[End of Your Opponent’s Turn]** | **End Phase** du tour adverse (10-2-8). |

### B.2 Activation en Main / Counter

| Marqueur | Signification |
|----------|----------------|
| **[Activate: Main]** | Effet activable en **Main Phase** (hors combat), par déclaration (10-2-2). |
| **[Main]** | Sur **Events** : activable depuis la main en Main Phase (hors combat) (10-2-3). |
| **[Counter]** | Sur **Events** : uniquement pendant le **Counter Step** adverse (10-2-4). |

### B.3 Conditions liées au DON!! et au tour

| Marqueur | Signification |
|----------|----------------|
| **[DON!! xX]** | Condition : au moins **X DON!! donnés** à cette carte (10-2-9 ; voir 8-3-2-3). |
| **DON!! −X** | Coût / condition : choisir **X DON!!** sur Leader, Characters et coût, et les **renvoyer** au deck DON!! (10-2-10). |
| **[Your Turn]** | Condition remplie **pendant votre tour** (10-2-11). |
| **[Opponent’s Turn]** | Condition remplie **pendant le tour adverse** (10-2-12). |
| **[Once Per Turn]** | L’effet ne peut être **activé et résolu qu’une fois** par tour (par carte) (10-2-13). |

### B.4 Autres mots-clés de règle

| Terme | Signification |
|--------|----------------|
| **K.O.** | Character **défaussé** après combat perdu ou par effet (10-2-1). |
| **Trash** (mot-clé) | Choisir une carte en **main** et la placer dans le **trash** (10-2-14). |

### B.5 *(Symbol) Counter* (section 2-10)

Sur les **Characters** : valeur de **bonus de puissance** activable pendant le **Counter Step** (contre une attaque). Ce n’est pas le même mot-clé que **[Counter]** sur les Events.

### B.6 **[Trigger]** sur la carte en zone Vie (section 2-11)

Fait partie du **texte de carte** ; permet d’activer un effet **à la place** d’ajouter la carte Vie à la main lors des dégâts (2-11).

---

## C. Actions de jeu fréquentes (sans crochets, section 4)

Utiles pour lire les effets « Draw », «Trash», «rest», etc. :

| Terme (règles EN) | Idée |
|-------------------|------|
| **Draw a card** (4-5) | Prendre la carte du dessus du **deck** en **main** sans la révéler à l’adversaire. |
| **Damage processing** (4-6) | Dégâts au Leader : déplacer des cartes **Vie → main** (sauf Trigger / effets). |
| **Play a card** (4-7) | Payer le **coût** et jouer / activer depuis la main. |
| **Active / Rested** (4-4) | Carte **verticale** = active, **horizontale** = reposée (sauf DON!! en zone coût). |

---

## D. Extraction dans `opctcg_text_sim`

- **`extract_keywords_from_text`** repère toutes les séquences **`[ … ]`**, **`{ … }`**, **`< … >`** dans l’ordre d’apparition (dédoublonnage).
- **`CardDef.keywords`** est rempli au chargement CSV à partir de **`card_text`** / **Description** tcgcsv.
- Le **manifeste** de replay et le **GIF** agrègent les marqueurs des cartes **visibles** (mains + terrain).

Le **simulateur texte** n’exécute pas encore la logique de chaque mot-clé : voir `OPTCG_GameRules_Reference.md` section *Cartographie simulateur*.
