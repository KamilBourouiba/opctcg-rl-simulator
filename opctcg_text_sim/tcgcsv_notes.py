"""
TCGCSV (https://tcgcsv.com) — miroir JSON de l’API TCGplayer (catégories, groupes, produits).

**One Piece Card Game** : ``categoryId`` = 68 (détecté automatiquement par ``find_one_piece_category_id``).

Pour remplir ``data/cards_tcgcsv.csv`` (images CDN, coût, puissance, counter, texte d’effet) ::

    python scripts/fetch_opcg_tcgcsv.py -o data/cards_tcgcsv.csv

Options utiles : ``--list-groups``, ``--groups 23589,24241`` (IDs de sets), ``--category-id 68``.

Voir ``opctcg_text_sim/tcgcsv_fetch.py`` pour l’accès programmatique.
"""
