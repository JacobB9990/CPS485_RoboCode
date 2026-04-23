package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;

public interface RadarController {
    void apply(AdvancedRobot robot, BotContext context, EnemySnapshot target);
}
