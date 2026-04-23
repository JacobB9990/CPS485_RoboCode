package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;

public interface MovementController {
    void apply(AdvancedRobot robot, BotContext context, TacticalMode mode, EnemySnapshot target);
}
