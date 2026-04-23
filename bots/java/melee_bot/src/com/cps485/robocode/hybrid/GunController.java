package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;

public interface GunController {
    void apply(AdvancedRobot robot, BotContext context, EnemySnapshot target, TacticalMode mode);
}
