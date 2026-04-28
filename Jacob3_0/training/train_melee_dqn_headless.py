from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bots.python.dqn.training.train_melee_dqn_headless import main


if __name__ == "__main__":
    main()
