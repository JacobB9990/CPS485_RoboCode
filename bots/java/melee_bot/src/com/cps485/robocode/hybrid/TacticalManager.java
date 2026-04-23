package com.cps485.robocode.hybrid;

public interface TacticalManager {
    TacticalMode chooseMode(BotContext context, EnemySnapshot target);
}
