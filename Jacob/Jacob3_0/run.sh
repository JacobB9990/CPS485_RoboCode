#!/bin/bash
# Run DQN bot with required environment variables

export BOT_NAME="DQNBot"
export BOT_VERSION="1.0.0"

python3 ./dqn_bot.py "$@"
