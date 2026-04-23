package cw;

final class EnemySnapshot {
    final String name;

    double x;
    double y;
    double distance;
    double absBearingRadians;
    double relBearingRadians;
    double energy;
    double velocity;
    double headingRadians;
    double lateralVelocity;
    long lastSeenTick;
    boolean alive = true;

    EnemySnapshot(String name) {
        this.name = name;
    }

    long age(long now) {
        return now - lastSeenTick;
    }
}
