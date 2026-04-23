package cw.meleedqn;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import robocode.AdvancedRobot;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;

public final class EnemyManager {
    private final Map<String, EnemySnapshot> enemies = new LinkedHashMap<>();

    public void update(AdvancedRobot robot, ScannedRobotEvent event, long tick) {
        EnemySnapshot snapshot = enemies.computeIfAbsent(event.getName(), EnemySnapshot::new);
        snapshot.update(robot, event, tick);
    }

    public void onRobotDeath(RobotDeathEvent event) {
        enemies.remove(event.getName());
    }

    public void resetRound() {
        enemies.clear();
    }

    public Collection<EnemySnapshot> all() {
        return enemies.values();
    }

    public int count() {
        return enemies.size();
    }

    public EnemySnapshot get(String name) {
        return enemies.get(name);
    }

    public EnemySnapshot nearest() {
        return enemies.values().stream()
                .min(Comparator.comparingDouble(EnemySnapshot::getDistance))
                .orElse(null);
    }

    public EnemySnapshot weakest() {
        return enemies.values().stream()
                .min(Comparator.comparingDouble(EnemySnapshot::getEnergy))
                .orElse(null);
    }

    public EnemySnapshot stalest(long tick) {
        return enemies.values().stream()
                .max(Comparator.comparingLong(enemy -> enemy.age(tick)))
                .orElse(null);
    }

    public double closestDistance() {
        EnemySnapshot nearest = nearest();
        return nearest == null ? 0.0 : nearest.getDistance();
    }

    public double averageDistance() {
        if (enemies.isEmpty()) {
            return 0.0;
        }
        double sum = 0.0;
        for (EnemySnapshot enemy : enemies.values()) {
            sum += enemy.getDistance();
        }
        return sum / enemies.size();
    }

    public List<EnemySnapshot> list() {
        return new ArrayList<>(enemies.values());
    }
}
