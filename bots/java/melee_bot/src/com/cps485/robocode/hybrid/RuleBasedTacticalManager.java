package com.cps485.robocode.hybrid;

public final class RuleBasedTacticalManager implements TacticalManager {
    @Override
    public TacticalMode chooseMode(BotContext context, EnemySnapshot target) {
        if (context.isCrowded()) {
            return TacticalMode.ESCAPE_CROWD;
        }
        if (context.isLowEnergy() && context.getOthers() > 1) {
            return TacticalMode.SURVIVE;
        }
        if (target != null && target.getEnergy() < 18.0 && target.getDistance() < 300.0) {
            return TacticalMode.FINISH_WEAK_TARGET;
        }
        if (context.isNearWall()) {
            return TacticalMode.REPOSITION;
        }
        return TacticalMode.ENGAGE;
    }
}
