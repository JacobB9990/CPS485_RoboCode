package com.cps485.robocode.hybrid;

import robocode.AdvancedRobot;
import robocode.Rules;
import robocode.util.Utils;

public final class GuessFactorGunController implements GunController {
    @Override
    public void apply(AdvancedRobot robot, BotContext context, EnemySnapshot target, TacticalMode mode) {
        if (target == null) {
            return;
        }

        double power = chooseFirePower(context, target, mode);
        double bulletSpeed = Rules.getBulletSpeed(power);
        double timeToTarget = target.getDistance() / bulletSpeed;
        double predictedX = target.getX() + Math.sin(target.getHeadingRadians()) * target.getVelocity() * timeToTarget;
        double predictedY = target.getY() + Math.cos(target.getHeadingRadians()) * target.getVelocity() * timeToTarget;

        predictedX = Math.max(18.0, Math.min(context.getBattlefieldWidth() - 18.0, predictedX));
        predictedY = Math.max(18.0, Math.min(context.getBattlefieldHeight() - 18.0, predictedY));

        double aimBearing = Geometry.absoluteBearing(context.getX(), context.getY(), predictedX, predictedY);
        double gunTurn = Utils.normalRelativeAngle(aimBearing - context.getGunHeadingRadians());
        robot.setTurnGunRightRadians(gunTurn);

        if (robot.getGunHeat() == 0.0 && Math.abs(robot.getGunTurnRemainingRadians()) < Math.toRadians(8.0)) {
            robot.setFire(power);
        }
    }

    public double chooseFirePower(BotContext context, EnemySnapshot target, TacticalMode mode) {
        double distanceFactor = clamp(450.0 / Math.max(120.0, target.getDistance()), 0.45, 2.2);
        double finishingBonus = target.getEnergy() < 16.0 ? 0.45 : 0.0;
        double survivalPenalty = mode == TacticalMode.SURVIVE ? 0.5 : 0.0;
        double crowdPenalty = context.isCrowded() ? 0.25 : 0.0;
        double energyBudget = clamp(context.getEnergy() / 35.0, 0.5, 1.5);
        double hitChanceProxy = clamp(1.4 - Math.abs(target.getVelocity()) / 8.0, 0.55, 1.15);

        double power = (1.1 * distanceFactor * hitChanceProxy * energyBudget)
                + finishingBonus
                - survivalPenalty
                - crowdPenalty;

        power = Math.min(power, target.getEnergy() / 4.0 + 0.5);
        power = Math.min(power, context.getEnergy() - 0.2);
        return clamp(power, 0.1, 3.0);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }
}
