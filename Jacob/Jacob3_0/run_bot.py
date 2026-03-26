"""Launcher for DQN bot with environment setup."""

import os
import sys

# Set required environment variables BEFORE importing robocode
os.environ.setdefault("BOT_NAME", "DQNBot")
os.environ.setdefault("BOT_VERSION", "1.0.0")


if __name__ == "__main__":
    from dqn_bot import main

    main()
