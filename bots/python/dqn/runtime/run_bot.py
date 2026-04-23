"""Launcher for the packaged DQN bot runtime."""

import os

# Set required environment variables BEFORE importing robocode
os.environ.setdefault("BOT_NAME", "DQNBot")
os.environ.setdefault("BOT_VERSION", "1.0.0")


if __name__ == "__main__":
    from bots.python.dqn.runtime.dqn_bot import main

    main()
