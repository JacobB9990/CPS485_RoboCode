package cw.meleedqn;

import robocode.AdvancedRobot;
import robocode.util.Utils;

public final class RewardTracker {
    private static final double DAMAGE_DEALT_COEF = 0.045;
    private static final double DAMAGE_TAKEN_COEF = -0.06;
    private static final double KILL_COEF = 1.2;
    private static final double SURVIVAL_TICK_COEF = 0.003;
    private static final double WALL_HIT_COEF = -0.18;
    private static final double ROBOT_COLLISION_COEF = -0.14;
    private static final double DENSE_ZONE_COEF = -0.08;
    private static final double INACTIVITY_COEF = -0.02;
    private static final double TARGET_SWITCH_COEF = -0.03;
    private static final double FIRE_COST_COEF = -0.005;
    private static final double ALIVE_FINISH_BONUS = 1.0;

    private double pendingReward;
    private double totalReward;
    private double damageDealt;
    private double damageTaken;
    private int kills;
    private long survivalTicks;
    private int recentlyHitTicks;
    private int repeatedStillTicks;
    private double lastX;
    private double lastY;
    private double lastHeading;
    private boolean initialized;

    public void resetRound() {
        pendingReward = 0.0;
        totalReward = 0.0;
        damageDealt = 0.0;
        damageTaken = 0.0;
        kills = 0;
        survivalTicks = 0L;
        recentlyHitTicks = 0;
        repeatedStillTicks = 0;
        lastX = 0.0;
        lastY = 0.0;
        lastHeading = 0.0;
        initialized = false;
    }

    public void onTick(AdvancedRobot robot, double crowdingScore, int targetSwitches) {
        survivalTicks++;
        pendingReward += SURVIVAL_TICK_COEF;
        pendingReward += DENSE_ZONE_COEF * crowdingScore;
        pendingReward += TARGET_SWITCH_COEF * targetSwitches;

        if (recentlyHitTicks > 0) {
            recentlyHitTicks--;
        }

        if (!initialized) {
            initialized = true;
            lastX = robot.getX();
            lastY = robot.getY();
            lastHeading = robot.getHeadingRadians();
            return;
        }

        double moved = Math.hypot(robot.getX() - lastX, robot.getY() - lastY);
        double turn = Math.abs(Utils.normalRelativeAngle(robot.getHeadingRadians() - lastHeading));
        if (moved < 4.0 && turn > 0.6) {
            repeatedStillTicks++;
        } else if (moved > 8.0) {
            repeatedStillTicks = Math.max(0, repeatedStillTicks - 2);
        } else {
            repeatedStillTicks = Math.max(0, repeatedStillTicks - 1);
        }

        if (repeatedStillTicks >= 6) {
            pendingReward += INACTIVITY_COEF;
        }

        lastX = robot.getX();
        lastY = robot.getY();
        lastHeading = robot.getHeadingRadians();
    }

    public void onBulletDamageDealt(double damage) {
        damageDealt += damage;
        pendingReward += DAMAGE_DEALT_COEF * damage;
    }

    public void onBulletDamageTaken(double damage) {
        damageTaken += damage;
        pendingReward += DAMAGE_TAKEN_COEF * damage;
        recentlyHitTicks = 8;
    }

    public void onKill() {
        kills++;
        pendingReward += KILL_COEF;
    }

    public void onHitWall() {
        pendingReward += WALL_HIT_COEF;
    }

    public void onRobotCollision() {
        pendingReward += ROBOT_COLLISION_COEF;
    }

    public void onFireCommand() {
        pendingReward += FIRE_COST_COEF;
    }

    public double consumeStepReward() {
        double clipped = Math.max(-2.0, Math.min(2.0, pendingReward));
        totalReward += clipped;
        pendingReward = 0.0;
        return clipped;
    }

    public double finishRound(int placement, int totalBots, boolean aliveAtEnd) {
        double placementRatio = totalBots <= 1 ? 1.0 : 1.0 - ((placement - 1.0) / (totalBots - 1.0));
        double finalReward = (1.4 * placementRatio) + (aliveAtEnd ? ALIVE_FINISH_BONUS : 0.0);
        pendingReward += finalReward;
        return consumeStepReward();
    }

    public boolean wasRecentlyHit() {
        return recentlyHitTicks > 0;
    }

    public double getTotalReward() {
        return totalReward;
    }

    public double getDamageDealt() {
        return damageDealt;
    }

    public double getDamageTaken() {
        return damageTaken;
    }

    public int getKills() {
        return kills;
    }

    public long getSurvivalTicks() {
        return survivalTicks;
    }
}
