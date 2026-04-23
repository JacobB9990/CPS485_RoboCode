package cw.meleedqn;

import java.util.Collection;

import robocode.AdvancedRobot;
import robocode.util.Utils;

public final class DangerMap {
    private static final double WALL_MARGIN = 80.0;

    public double crowdingScore(AdvancedRobot robot, Collection<EnemySnapshot> enemies) {
        if (enemies.isEmpty()) {
            return 0.0;
        }

        double danger = 0.0;
        for (EnemySnapshot enemy : enemies) {
            double dist = Math.max(36.0, enemy.getDistance());
            danger += 1.0 / (dist * dist);
        }

        return clamp01(danger * 25000.0);
    }

    public int countWithin(Collection<EnemySnapshot> enemies, double range) {
        int count = 0;
        for (EnemySnapshot enemy : enemies) {
            if (enemy.getDistance() <= range) {
                count++;
            }
        }
        return count;
    }

    public double safestHeading(AdvancedRobot robot, Collection<EnemySnapshot> enemies) {
        double bestHeading = robot.getHeadingRadians();
        double bestScore = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < 32; i++) {
            double heading = (2.0 * Math.PI * i) / 32.0;
            double score = headingScore(robot, enemies, heading);
            if (score > bestScore) {
                bestScore = score;
                bestHeading = heading;
            }
        }

        return bestHeading;
    }

    public double escapeHeading(AdvancedRobot robot, Collection<EnemySnapshot> enemies) {
        if (enemies.isEmpty()) {
            return safestHeading(robot, enemies);
        }

        double vx = 0.0;
        double vy = 0.0;
        for (EnemySnapshot enemy : enemies) {
            double dx = robot.getX() - enemy.getX();
            double dy = robot.getY() - enemy.getY();
            double distSq = Math.max(1600.0, dx * dx + dy * dy);
            vx += dx / distSq;
            vy += dy / distSq;
        }

        vx += wallRepulsion(robot.getX(), robot.getBattleFieldWidth());
        vy += wallRepulsion(robot.getY(), robot.getBattleFieldHeight());
        return Math.atan2(vx, vy);
    }

    public boolean forwardUnsafe(AdvancedRobot robot) {
        double heading = robot.getHeadingRadians();
        double nextX = robot.getX() + Math.sin(heading) * 120.0;
        double nextY = robot.getY() + Math.cos(heading) * 120.0;
        return nearWall(nextX, nextY, robot);
    }

    private double headingScore(AdvancedRobot robot, Collection<EnemySnapshot> enemies, double heading) {
        double probeX = robot.getX() + Math.sin(heading) * 140.0;
        double probeY = robot.getY() + Math.cos(heading) * 140.0;

        if (nearWall(probeX, probeY, robot)) {
            return -10.0;
        }

        double score = 0.0;
        for (EnemySnapshot enemy : enemies) {
            double dx = probeX - enemy.getX();
            double dy = probeY - enemy.getY();
            double distSq = Math.max(1600.0, dx * dx + dy * dy);
            score += Math.log(distSq);
        }

        double turnPenalty = Math.abs(Utils.normalRelativeAngle(heading - robot.getHeadingRadians()));
        return score - 0.75 * turnPenalty;
    }

    private boolean nearWall(double x, double y, AdvancedRobot robot) {
        return x < WALL_MARGIN
                || y < WALL_MARGIN
                || x > robot.getBattleFieldWidth() - WALL_MARGIN
                || y > robot.getBattleFieldHeight() - WALL_MARGIN;
    }

    private double wallRepulsion(double value, double maxValue) {
        double low = Math.max(1.0, value - WALL_MARGIN);
        double high = Math.max(1.0, maxValue - WALL_MARGIN - value);
        return (1.0 / low) - (1.0 / high);
    }

    private double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }
}
