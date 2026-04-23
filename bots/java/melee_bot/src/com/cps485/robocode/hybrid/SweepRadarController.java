package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;
import robocode.util.Utils;

public final class SweepRadarController implements RadarController {
    @Override
    public void apply(AdvancedRobot robot, BotContext context, EnemySnapshot target) {
        if (target == null) {
            robot.setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
            return;
        }

        double radarTurn = Utils.normalRelativeAngle(
                target.getAbsoluteBearingRadians() - context.getRadarHeadingRadians());
        double extraTurn = Math.atan(36.0 / Math.max(1.0, target.getDistance()));
        radarTurn += (radarTurn >= 0 ? extraTurn : -extraTurn);
        robot.setTurnRadarRightRadians(radarTurn);
    }
}
