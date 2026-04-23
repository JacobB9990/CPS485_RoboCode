package cw.meleedqn;

import robocode.AdvancedRobot;
import robocode.ScannedRobotEvent;
import robocode.util.Utils;

public final class EnemySnapshot {
    private final String name;
    private double x;
    private double y;
    private double distance;
    private double bearingRadians;
    private double headingRadians;
    private double velocity;
    private double energy;
    private long lastSeenTick;
    private double lastEnergyDrop;

    public EnemySnapshot(String name) {
        this.name = name;
    }

    public void update(AdvancedRobot robot, ScannedRobotEvent event, long tick) {
        double previousEnergy = energy;
        double absoluteBearing = robot.getHeadingRadians() + event.getBearingRadians();

        distance = event.getDistance();
        bearingRadians = event.getBearingRadians();
        headingRadians = event.getHeadingRadians();
        velocity = event.getVelocity();
        energy = event.getEnergy();
        x = robot.getX() + Math.sin(absoluteBearing) * distance;
        y = robot.getY() + Math.cos(absoluteBearing) * distance;
        lastSeenTick = tick;
        lastEnergyDrop = previousEnergy > 0.0 ? Math.max(0.0, previousEnergy - energy) : 0.0;
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

    public double getDistance() {
        return distance;
    }

    public double getBearingRadians() {
        return bearingRadians;
    }

    public double getHeadingRadians() {
        return headingRadians;
    }

    public double getVelocity() {
        return velocity;
    }

    public double getEnergy() {
        return energy;
    }

    public long getLastSeenTick() {
        return lastSeenTick;
    }

    public double getLastEnergyDrop() {
        return lastEnergyDrop;
    }

    public long age(long currentTick) {
        return Math.max(0L, currentTick - lastSeenTick);
    }

    public double absoluteBearingFrom(AdvancedRobot robot) {
        return Math.atan2(x - robot.getX(), y - robot.getY());
    }

    public double gunTurnFrom(AdvancedRobot robot) {
        return Utils.normalRelativeAngle(absoluteBearingFrom(robot) - robot.getGunHeadingRadians());
    }

    public double headingTowardRobotError(AdvancedRobot robot) {
        double toRobot = Math.atan2(robot.getX() - x, robot.getY() - y);
        return Math.abs(Utils.normalRelativeAngle(toRobot - headingRadians));
    }
}
