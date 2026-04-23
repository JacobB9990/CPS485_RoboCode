package com.cps485.robocode.hybrid;

public final class EnemySnapshot {
    private final String name;
    private final double x;
    private final double y;
    private final double energy;
    private final double headingRadians;
    private final double velocity;
    private final double distance;
    private final double absoluteBearingRadians;
    private final long lastSeenTick;
    private final boolean alive;

    public EnemySnapshot(
            String name,
            double x,
            double y,
            double energy,
            double headingRadians,
            double velocity,
            double distance,
            double absoluteBearingRadians,
            long lastSeenTick,
            boolean alive) {
        this.name = name;
        this.x = x;
        this.y = y;
        this.energy = energy;
        this.headingRadians = headingRadians;
        this.velocity = velocity;
        this.distance = distance;
        this.absoluteBearingRadians = absoluteBearingRadians;
        this.lastSeenTick = lastSeenTick;
        this.alive = alive;
    }

    public String getName() {
        return name;
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

    public double getHeadingRadians() {
        return headingRadians;
    }

    public double getVelocity() {
        return velocity;
    }

    public double getDistance() {
        return distance;
    }

    public double getAbsoluteBearingRadians() {
        return absoluteBearingRadians;
    }

    public long getLastSeenTick() {
        return lastSeenTick;
    }

    public boolean isAlive() {
        return alive;
    }

    public long age(long currentTick) {
        return Math.max(0L, currentTick - lastSeenTick);
    }

    public EnemySnapshot markDead() {
        return new EnemySnapshot(
                name,
                x,
                y,
                energy,
                headingRadians,
                velocity,
                distance,
                absoluteBearingRadians,
                lastSeenTick,
                false);
    }
}
