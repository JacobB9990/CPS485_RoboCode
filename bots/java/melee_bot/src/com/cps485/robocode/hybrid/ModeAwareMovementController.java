package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;
import robocode.util.Utils;

public final class ModeAwareMovementController implements MovementController {
    private int strafeDirection = 1;

    @Override
    public void apply(AdvancedRobot robot, BotContext context, TacticalMode mode, EnemySnapshot target) {
        if (mode == TacticalMode.ESCAPE_CROWD || mode == TacticalMode.REPOSITION) {
            moveToSafestCell(robot, context);
            return;
        }

        if (target == null) {
            robot.setAhead(100.0);
            robot.setTurnRightRadians(0.4);
            return;
        }

        switch (mode) {
            case SURVIVE:
                orbit(robot, context, target, 225.0, 0.9);
                break;
            case FINISH_WEAK_TARGET:
                orbit(robot, context, target, 160.0, 0.5);
                break;
            case ENGAGE:
            default:
                orbit(robot, context, target, 250.0, 0.7);
                break;
        }
    }

    public void reverseDirection() {
        strafeDirection *= -1;
    }

    private void moveToSafestCell(AdvancedRobot robot, BotContext context) {
        DangerMap.Point2 safest = context.getDangerMap().pickSafestCellCenter(context.getX(), context.getY());
        double angle = Geometry.absoluteBearing(context.getX(), context.getY(), safest.x, safest.y);
        double turn = Geometry.normalRelativeAngle(angle - context.getHeadingRadians());
        double distance = Geometry.distance(context.getX(), context.getY(), safest.x, safest.y);
        setBackAsFront(robot, turn, Math.min(160.0, distance));
    }

    private void orbit(AdvancedRobot robot, BotContext context, EnemySnapshot target, double preferredRange, double aggressiveness) {
        double absoluteBearing = target.getAbsoluteBearingRadians();
        double offset = (Math.PI / 2.0) * strafeDirection;
        double desiredHeading = absoluteBearing + offset;
        double distanceError = target.getDistance() - preferredRange;
        double turn = Utils.normalRelativeAngle(desiredHeading - context.getHeadingRadians());
        robot.setTurnRightRadians(turn);
        robot.setAhead(120.0 + (distanceError * aggressiveness));
    }

    private void setBackAsFront(AdvancedRobot robot, double angle, double distance) {
        double normalized = Utils.normalRelativeAngle(angle);
        if (Math.abs(normalized) > Math.PI / 2.0) {
            double turn = Utils.normalRelativeAngle(normalized + Math.PI);
            robot.setTurnRightRadians(turn);
            robot.setBack(distance);
        } else {
            robot.setTurnRightRadians(normalized);
            robot.setAhead(distance);
        }
    }
}
