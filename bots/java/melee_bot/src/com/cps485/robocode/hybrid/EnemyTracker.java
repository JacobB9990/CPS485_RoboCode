package com.cps485.robocode.hybrid;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import robocode.AdvancedRobot;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;

public final class EnemyTracker {
    private static final long STALE_TICKS = 40L;

    private final Map<String, EnemySnapshot> enemies = new LinkedHashMap<>();

    public void onScannedRobot(AdvancedRobot robot, ScannedRobotEvent event) {
        double absoluteBearing = robot.getHeadingRadians() + event.getBearingRadians();
        double enemyX = robot.getX() + Math.sin(absoluteBearing) * event.getDistance();
        double enemyY = robot.getY() + Math.cos(absoluteBearing) * event.getDistance();

        EnemySnapshot snapshot = new EnemySnapshot(
                event.getName(),
                enemyX,
                enemyY,
                event.getEnergy(),
                event.getHeadingRadians(),
                event.getVelocity(),
                event.getDistance(),
                absoluteBearing,
                robot.getTime(),
                true);
        enemies.put(event.getName(), snapshot);
    }

    public void onRobotDeath(RobotDeathEvent event) {
        EnemySnapshot snapshot = enemies.get(event.getName());
        if (snapshot != null) {
            enemies.put(event.getName(), snapshot.markDead());
        }
    }

    public List<EnemySnapshot> getAliveEnemies(long currentTick) {
        List<EnemySnapshot> result = new ArrayList<>();
        for (EnemySnapshot enemy : enemies.values()) {
            if (enemy.isAlive() && enemy.age(currentTick) <= STALE_TICKS) {
                result.add(enemy);
            }
        }
        result.sort(Comparator.comparingDouble(EnemySnapshot::getDistance));
        return result;
    }

    public int countNearby(double x, double y, double radius, long currentTick) {
        int count = 0;
        double radiusSq = radius * radius;
        for (EnemySnapshot enemy : getAliveEnemies(currentTick)) {
            double dx = enemy.getX() - x;
            double dy = enemy.getY() - y;
            if ((dx * dx) + (dy * dy) <= radiusSq) {
                count++;
            }
        }
        return count;
    }

    public Collection<EnemySnapshot> getAllEnemies() {
        return enemies.values();
    }
}
