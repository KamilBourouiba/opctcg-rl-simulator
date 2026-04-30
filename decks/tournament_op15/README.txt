Decks du tournoi Egman (PEC Occitania Lille Regionals, OP15) — générés automatiquement.

Pour régénérer depuis l’URL du tournoi (paramètre t=) :
  python3 scripts/fetch_egman_tournament_decks.py --tournament-url "https://deckbuilder.egmanevents.com/optcg/tournaments?format=OP15&t=009ab10b-49d1-4b3e-9342-5b6eb808c15a"

Ou avec l’UUID seul :
  python3 scripts/fetch_egman_tournament_decks.py --tournament-id 009ab10b-49d1-4b3e-9342-5b6eb808c15a

Les listes viennent de la table Supabase ``tournament_results`` (champ ``deck_list_url``). Ce dossier est écrasé à chaque fetch (sauf ce README).

Ensuite gauntlet / site :
  python3 scripts/deck_gauntlet_web.py --decks-dir decks/tournament_op15
