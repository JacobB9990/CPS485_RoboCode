package cw.meleedqn;

import java.util.Collection;

import robocode.AdvancedRobot;

public final class TargetSelector {
    private static final double SWITCH_MARGIN = 0.18;
    private static final long MIN_TARGET_HOLD_TICKS = 10L;

    private String currentTargetName;
    private long lastSwitchTick;
    private int switchCount;

    public EnemySnapshot select(AdvancedRobot robot, Collection<EnemySnapshot> enemies, long tick) {
        EnemySnapshot best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (EnemySnapshot enemy : enemies) {
            double score = score(robot, enemy, tick);
            if (score > bestScore) {
                bestScore = score;
                best = enemy;
            }
        }

        EnemySnapshot current = currentTargetName == null ? null : findByName(enemies, currentTargetName);
        if (current == null && best != null) {
            currentTargetName = best.getName();
            lastSwitchTick = tick;
            return best;
        }

        if (current == null) {
            return null;
        }

        if (best == null) {
            currentTargetName = null;
            return null;
        }

        double currentScore = score(robot, current, tick);
        boolean shouldSwitch = !best.getName().equals(current.getName())
                && (bestScore > currentScore + SWITCH_MARGIN)
                && (tick - lastSwitchTick >= MIN_TARGET_HOLD_TICKS);

        if (shouldSwitch) {
            currentTargetName = best.getName();
            lastSwitchTick = tick;
            switchCount++;
            return best;
        }

        return current;
    }

    public double score(AdvancedRobot robot, EnemySnapshot enemy, long tick) {
        double distanceScore = 1.0 - clamp01(enemy.getDistance() / 900.0);
        double lowEnergyScore = 1.0 - clamp01(enemy.getEnergy() / 100.0);
        double freshnessScore = 1.0 - clamp01(enemy.age(tick) / 40.0);
        double gunEaseScore = 1.0 - clamp01(Math.abs(enemy.gunTurnFrom(robot)) / Math.PI);

        double fireIntentScore = 0.0;
        if (enemy.getLastEnergyDrop() >= 0.1 && enemy.getLastEnergyDrop() <= 3.0) {
            fireIntentScore += 0.55;
        }
        if (enemy.headingTowardRobotError(robot) < Math.toRadians(22.0)) {
            fireIntentScore += 0.45;
        }

        double score = 0.34 * distanceScore
                + 0.22 * lowEnergyScore
                + 0.15 * freshnessScore
                + 0.15 * fireIntentScore
                + 0.14 * gunEaseScore;

        if (enemy.getName().equals(currentTargetName)) {
            score += 0.08;
        }
        return score;
    }

    public EnemySnapshot mostThreatening(AdvancedRobot robot, Collection<EnemySnapshot> enemies, long tick) {
        EnemySnapshot best = null;
        double bestThreat = Double.NEGATIVE_INFINITY;
        for (EnemySnapshot enemy : enemies) {
            double threat = 0.55 * (1.0 - clamp01(enemy.getDistance() / 750.0))
                    + 0.25 * clamp01(enemy.getEnergy() / 100.0)
                    + 0.20 * clamp01(score(robot, enemy, tick));
            if (threat > bestThreat) {
                bestThreat = threat;
                best = enemy;
            }
        }
        return best;
    }

    public int getSwitchCount() {
        return switchCount;
    }

    public String getCurrentTargetName() {
        return currentTargetName;
    }

    public void resetRound() {
        currentTargetName = null;
        lastSwitchTick = 0L;
        switchCount = 0;
    }

    private EnemySnapshot findByName(Collection<EnemySnapshot> enemies, String name) {
        for (EnemySnapshot enemy : enemies) {
            if (enemy.getName().equals(name)) {
                return enemy;
            }
        }
        return null;
    }

    private double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
