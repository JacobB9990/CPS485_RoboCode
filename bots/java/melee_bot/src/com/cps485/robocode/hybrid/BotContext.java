package com.cps485.robocode.hybrid;

import java.util.List;

public final class BotContext {
    private final long time;
    private final double x;
    private final double y;
    private final double energy;
    private final double velocity;
    private final double headingRadians;
    private final double gunHeadingRadians;
    private final double radarHeadingRadians;
    private final double battlefieldWidth;
    private final double battlefieldHeight;
    private final int others;
    private final List<EnemySnapshot> enemies;
    private final EnemyTracker enemyTracker;
    private final DangerMap dangerMap;

    public BotContext(
            long time,
            double x,
            double y,
            double energy,
            double velocity,
            double headingRadians,
            double gunHeadingRadians,
            double radarHeadingRadians,
            double battlefieldWidth,
            double battlefieldHeight,
            int others,
            List<EnemySnapshot> enemies,
            EnemyTracker enemyTracker,
            DangerMap dangerMap) {
        this.time = time;
        this.x = x;
        this.y = y;
        this.energy = energy;
        this.velocity = velocity;
        this.headingRadians = headingRadians;
        this.gunHeadingRadians = gunHeadingRadians;
        this.radarHeadingRadians = radarHeadingRadians;
        this.battlefieldWidth = battlefieldWidth;
        this.battlefieldHeight = battlefieldHeight;
        this.others = others;
        this.enemies = enemies;
        this.enemyTracker = enemyTracker;
        this.dangerMap = dangerMap;
    }

    public long getTime() {
        return time;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getEnergy() {
        return energy;
    }

    public double getVelocity() {
        return velocity;
    }

    public double getHeadingRadians() {
        return headingRadians;
    }

    public double getGunHeadingRadians() {
        return gunHeadingRadians;
    }

    public double getRadarHeadingRadians() {
        return radarHeadingRadians;
    }

    public double getBattlefieldWidth() {
        return battlefieldWidth;
    }

    public double getBattlefieldHeight() {
        return battlefieldHeight;
    }

    public int getOthers() {
        return others;
    }

    public List<EnemySnapshot> getEnemies() {
        return enemies;
    }

    public EnemyTracker getEnemyTracker() {
        return enemyTracker;
    }

    public DangerMap getDangerMap() {
        return dangerMap;
    }

    public boolean isLowEnergy() {
        return energy < 25.0;
    }

    public boolean isCrowded() {
        return others >= 4 || enemyTracker.countNearby(x, y, 225.0, time) >= 3;
    }

    public boolean isNearWall() {
        return x < 70.0
                || y < 70.0
                || x > battlefieldWidth - 70.0
                || y > battlefieldHeight - 70.0;
    }
}
