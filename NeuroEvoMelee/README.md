# NeuroEvo Melee Bot (Python)

This bot is a Python Tank Royale melee bot whose policy is a fixed-topology neural network evolved externally with a genetic algorithm.

## Runtime contract

- Bot directory: `NeuroEvoMelee`
- Metadata: `NeuroEvoMeleeBot.json`
- Launcher: `NeuroEvoMeleeBot.sh`
- Policy genome file path comes from `NEURO_GENOME_PATH`
- Optional telemetry JSONL path comes from `NEURO_TELEMETRY_PATH`

## State inputs (fixed-size)

The bot encodes exactly 20 inputs each tick:

- 1. self energy
- 2. self velocity
- 3. heading sin
- 4. heading cos
- 5. wall proximity
- 6. nearest distance
- 7. nearest bearing sin
- 8. nearest bearing cos
- 9. nearest energy
- 10. weakest distance
- 11. weakest bearing sin
- 12. weakest bearing cos
- 13. weakest energy
- 14. threat sum feature
- 15. max threat feature
- 16. crowd density feature
- 17. enemies alive
- 18. gun heat
- 19. target alignment
- 20. target preference delta feature

## Outputs

The evolved network produces 4 outputs:

1. movement turn command
2. movement distance command (forward/back)
3. fire intensity (also acts as fire/no-fire threshold)
4. target preference adjustment (nearest vs weakest bias)

## Python modules

- `runtime/neuroevo_melee_bot.py`: bot lifecycle, events, action application
- `genome/feature_encoder.py`: fixed-size multi-enemy feature extraction
- `genome/enemy_state.py`: tracked enemy memory record
- `genome/genome_loader.py`: lightweight genome JSON loader
- `genome/genome_network.py`: feedforward inference for evolved weights

## Training

Run the genetic trainer with the default local evaluation heuristic:

```bash
python3 -m NeuroEvoMelee.training.train_neuroevo_melee
```

Or point it at an external battle/evaluation command that prints a fitness value:

```bash
python3 -m NeuroEvoMelee.training.train_neuroevo_melee \
	--evaluate-command './NeuroEvoMeleeBot.sh --genome {genome}'
```
