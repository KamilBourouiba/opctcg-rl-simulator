A/B ressources (CPU / RAM / workers)
===================================

1) Fusion YAML
   Les petits fichiers ici s’ajoutent par-dessus config.yaml :

   .venv/bin/python scripts/train_ppo.py --config config.yaml \
     --config-overlay configs/ab/a_few_workers_more_ram.yaml \
     --gauntlet-decks-dir decks/tournament_op15 --steps 500000 --n-envs 0

   --n-envs 0 : respecter n_envs du YAML (sinon la CLI écrase l’overlay).

2) Clés optionnelles (dans training, ou via overlay)
   - main_torch_num_threads : limite torch.set_num_threads côté process principal
   - main_torch_interop_threads : limite torch.set_num_interop_threads

3) Script deux runs + dernier stp_s dans metrics
   AB_STEPS=400000 ./scripts/run_ab_benchmark.sh
