package cw;

import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;
import robocode.StatusEvent;
import robocode.util.Utils;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Melee-focused AdvancedRobot that maintains an online model per opponent and adapts
 * targeting plus movement using a threat score.
 */
public class MeleeOpponentModelBot extends AdvancedRobot {

    private static final double WALL_MARGIN = 80.0;
    private static final double MAX_TRACK_AGE = 45.0;
    private static final double DISENGAGE_DISTANCE = 180.0;
    private static final double RADAR_OVERSCAN = Math.toRadians(18.0);
    private static final double MOVEMENT_STEP = 140.0;

    private final Map<String, EnemyModel> enemies = new HashMap<>();
    private final AdaptivePolicy adaptivePolicy = new AdaptivePolicy();
    private Point2D.Double myLocation = new Point2D.Double();
    private EnemyModel currentTarget;
    private int moveDirection = 1;
    private long lastHitByBulletTime = -1;

    @Override
    public void run() {
        setBodyColor(new Color(36, 52, 71));
        setGunColor(new Color(211, 94, 96));
        setRadarColor(new Color(247, 197, 72));
        setBulletColor(new Color(247, 197, 72));
        setScanColor(new Color(153, 216, 201));

        setAdjustRadarForGunTurn(true);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForRobotTurn(true);
        setTurnRadarRightRadians(Double.POSITIVE_INFINITY);

        while (true) {
            myLocation = new Point2D.Double(getX(), getY());
            refreshThreats();
            adaptivePolicy.update(liveEnemies());
            currentTarget = chooseTarget();
            doMovement();
            doGunAndFire();
            doRadar();
            execute();
        }
    }

    @Override
    public void onStatus(StatusEvent event) {
        myLocation = new Point2D.Double(getX(), getY());
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        EnemyModel model = enemies.computeIfAbsent(event.getName(), EnemyModel::new);
        model.updateFromScan(this, event, myLocation);
    }

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        EnemyModel model = enemies.get(event.getName());
        if (model != null) {
            model.alive = false;
        }
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        lastHitByBulletTime = getTime();
        EnemyModel shooter = guessShooter(event.getBearingRadians());
        if (shooter != null) {
            shooter.registerHitOnMe();
        }
        moveDirection *= -1;
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        EnemyModel model = enemies.get(event.getName());
        if (model != null) {
            model.registerMyHit();
        }
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        EnemyModel model = enemies.get(event.getName());
        if (model != null) {
            model.closeContactCount++;
        }
        moveDirection *= -1;
        setBack(80);
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        moveDirection *= -1;
        setTurnRightRadians(normalizeBearing(Math.PI / 2 - event.getBearingRadians()));
        setAhead(120);
    }

    private void refreshThreats() {
        for (EnemyModel model : enemies.values()) {
            if (!model.alive) {
                continue;
            }
            model.refreshDerivedMetrics();
        }
    }

    private EnemyModel chooseTarget() {
        Collection<EnemyModel> alive = liveEnemies();
        if (alive.isEmpty()) {
            return null;
        }

        EnemyModel best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (EnemyModel enemy : alive) {
            double freshness = clamp(1.0 - ((getTime() - enemy.lastSeenTime) / MAX_TRACK_AGE), 0.0, 1.0);
            if (freshness <= 0.0) {
                continue;
            }

            double distanceFactor = 1.0 - clamp(enemy.distance / 900.0, 0.0, 1.0);
            double accuracyWindow = 1.0 - clamp(Math.abs(enemy.lateralVelocity) / 8.0, 0.0, 1.0);
            double lowEnergyFinish = 1.0 - clamp(enemy.energy / 100.0, 0.0, 1.0);
            double score =
                enemy.threatScore * 0.45 +
                distanceFactor * 0.22 +
                accuracyWindow * 0.18 +
                lowEnergyFinish * 0.10 +
                freshness * 0.05;

            if (enemy.category == EnemyCategory.SPINNER_WEAK_BOT) {
                score += 0.08;
            }

            if (score > bestScore) {
                bestScore = score;
                best = enemy;
            }
        }
        return best;
    }

    private void doGunAndFire() {
        if (currentTarget == null || getOthers() == 0) {
            return;
        }

        double firePower = chooseFirePower(currentTarget);
        double bulletSpeed = 20.0 - 3.0 * firePower;
        Point2D.Double predicted = predictLinearPosition(currentTarget, bulletSpeed);
        double gunAngle = absoluteBearing(myLocation, predicted);
        setTurnGunRightRadians(normalizeBearing(gunAngle - getGunHeadingRadians()));

        boolean aimed = Math.abs(getGunTurnRemainingRadians()) < Math.toRadians(6.0);
        if (aimed && getGunHeat() == 0.0 && getEnergy() > firePower) {
            setFire(firePower);
            currentTarget.registerMyShot();
        }
    }

    private double chooseFirePower(EnemyModel target) {
        double power = 1.4;
        if (target.distance < 200) {
            power += 0.8;
        } else if (target.distance < 350) {
            power += 0.4;
        }
        if (target.threatScore > 0.75) {
            power += 0.4;
        }
        if (getOthers() > 3) {
            power -= 0.3;
        }
        if (getEnergy() < 25) {
            power -= 0.4;
        }
        return clamp(power, 0.8, 2.8);
    }

    private void doMovement() {
        Collection<EnemyModel> alive = liveEnemies();
        if (alive.isEmpty()) {
            setAhead(120 * moveDirection);
            return;
        }

        EnemyModel topThreat = alive.stream()
            .max(Comparator.comparingDouble(enemy -> enemy.threatScore))
            .orElse(null);
        boolean shouldDisengage =
            topThreat != null &&
            topThreat.distance < DISENGAGE_DISTANCE + adaptivePolicy.disengageBias * 70.0 &&
            (topThreat.threatScore > (0.82 - adaptivePolicy.disengageBias * 0.2) || getEnergy() < topThreat.energy);

        List<Double> candidates = new ArrayList<>();
        double baseAngle = (topThreat != null ? topThreat.absoluteBearing : getHeadingRadians()) + (Math.PI / 2.0 * moveDirection);
        candidates.add(baseAngle);
        candidates.add(baseAngle + Math.toRadians(30));
        candidates.add(baseAngle - Math.toRadians(30));
        candidates.add(baseAngle + Math.toRadians(60));
        candidates.add(baseAngle - Math.toRadians(60));
        if (shouldDisengage && topThreat != null) {
            candidates.add(topThreat.absoluteBearing + Math.PI);
        }

        double bestAngle = candidates.get(0);
        double bestDanger = Double.POSITIVE_INFINITY;
        for (double angle : candidates) {
            double smoothed = wallSmooth(myLocation, angle, moveDirection);
            Point2D.Double destination = project(myLocation, smoothed, MOVEMENT_STEP);
            double danger = dangerAt(destination, alive);
            if (danger < bestDanger) {
                bestDanger = danger;
                bestAngle = smoothed;
            }
        }

        goTo(project(myLocation, bestAngle, MOVEMENT_STEP));
        if (shouldDisengage || (lastHitByBulletTime >= 0 && getTime() - lastHitByBulletTime < 8)) {
            moveDirection *= -1;
        }
    }

    private void doRadar() {
        EnemyModel radarTarget = currentTarget;
        if (radarTarget == null) {
            setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
            return;
        }
        double radarTurn = normalizeBearing(radarTarget.absoluteBearing - getRadarHeadingRadians());
        double extra = radarTurn >= 0 ? RADAR_OVERSCAN : -RADAR_OVERSCAN;
        setTurnRadarRightRadians(radarTurn + extra);
    }

    private double dangerAt(Point2D.Double destination, Collection<EnemyModel> alive) {
        double danger = 0.0;

        for (EnemyModel enemy : alive) {
            double distance = Math.max(36.0, destination.distance(enemy.x, enemy.y));
            double clusterFactor = 1.0 + (enemy.nearbyEnemyCount * 0.12);
            double directness = 1.0 - Math.abs(Math.cos(absoluteBearing(new Point2D.Double(enemy.x, enemy.y), destination) - enemy.absoluteBearing));
            double ramPenalty = enemy.category == EnemyCategory.CLOSE_RANGE_AGGRESSOR
                ? adaptivePolicy.antiRamBias * clamp(1.0 - distance / 220.0, 0.0, 1.0)
                : 0.0;
            danger += enemy.threatScore * clusterFactor * (65000.0 / (distance * distance)) * (0.65 + 0.35 * directness + ramPenalty);
        }

        double edgePenalty = 0.0;
        edgePenalty += Math.max(0.0, WALL_MARGIN - destination.x) * 0.025;
        edgePenalty += Math.max(0.0, WALL_MARGIN - destination.y) * 0.025;
        edgePenalty += Math.max(0.0, destination.x - (getBattleFieldWidth() - WALL_MARGIN)) * 0.025;
        edgePenalty += Math.max(0.0, destination.y - (getBattleFieldHeight() - WALL_MARGIN)) * 0.025;

        Point2D.Double center = new Point2D.Double(getBattleFieldWidth() / 2.0, getBattleFieldHeight() / 2.0);
        double centerDistance = destination.distance(center);
        double centerPenalty = adaptivePolicy.centerAversion * clamp(1.0 - centerDistance / 260.0, 0.0, 1.0);

        return danger + edgePenalty + centerPenalty;
    }

    private EnemyModel guessShooter(double bulletBearingRadians) {
        double absoluteBulletBearing = getHeadingRadians() + bulletBearingRadians + Math.PI;
        EnemyModel best = null;
        double bestDiff = Double.POSITIVE_INFINITY;

        for (EnemyModel enemy : liveEnemies()) {
            double diff = Math.abs(normalizeBearing(enemy.absoluteBearing - absoluteBulletBearing));
            if (diff < bestDiff) {
                bestDiff = diff;
                best = enemy;
            }
        }
        return bestDiff < Math.toRadians(20.0) ? best : null;
    }

    private Collection<EnemyModel> liveEnemies() {
        List<EnemyModel> alive = new ArrayList<>();
        for (EnemyModel enemy : enemies.values()) {
            if (enemy.alive) {
                alive.add(enemy);
            }
        }

        for (EnemyModel enemy : alive) {
            int nearby = 0;
            for (EnemyModel other : alive) {
                if (enemy == other) {
                    continue;
                }
                if (Point2D.distance(enemy.x, enemy.y, other.x, other.y) < 220.0) {
                    nearby++;
                }
            }
            enemy.nearbyEnemyCount = nearby;
        }
        return alive;
    }

    private Point2D.Double predictLinearPosition(EnemyModel enemy, double bulletSpeed) {
        Point2D.Double predicted = new Point2D.Double(enemy.x, enemy.y);
        double heading = enemy.headingRadians;
        double velocity = enemy.velocity;
        long ticks = 0;

        while ((++ticks) * bulletSpeed < myLocation.distance(predicted)) {
            predicted = project(predicted, heading, velocity);
            predicted.x = clamp(predicted.x, WALL_MARGIN, getBattleFieldWidth() - WALL_MARGIN);
            predicted.y = clamp(predicted.y, WALL_MARGIN, getBattleFieldHeight() - WALL_MARGIN);
        }
        return predicted;
    }

    private void goTo(Point2D.Double destination) {
        double angle = normalizeBearing(absoluteBearing(myLocation, destination) - getHeadingRadians());
        double turn = Math.atan(Math.tan(angle));
        setTurnRightRadians(turn);
        if (angle == turn) {
            setAhead(myLocation.distance(destination));
        } else {
            setBack(myLocation.distance(destination));
        }
    }

    private double wallSmooth(Point2D.Double source, double angle, int direction) {
        Point2D.Double test = project(source, angle, WALL_MARGIN);
        int guard = 0;
        while (!inSafeField(test) && guard++ < 25) {
            angle += direction * 0.12;
            test = project(source, angle, WALL_MARGIN);
        }
        return angle;
    }

    private boolean inSafeField(Point2D.Double point) {
        return point.x > WALL_MARGIN &&
            point.y > WALL_MARGIN &&
            point.x < getBattleFieldWidth() - WALL_MARGIN &&
            point.y < getBattleFieldHeight() - WALL_MARGIN;
    }

    private static Point2D.Double project(Point2D.Double source, double angle, double length) {
        return new Point2D.Double(
            source.x + Math.sin(angle) * length,
            source.y + Math.cos(angle) * length
        );
    }

    private static double absoluteBearing(Point2D.Double source, Point2D.Double target) {
        return Math.atan2(target.x - source.x, target.y - source.y);
    }

    private static double normalizeBearing(double angle) {
        return Utils.normalRelativeAngle(angle);
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private enum MovementStyle {
        SPINNER,
        STRAFER,
        RAMMER,
        CAMPER,
        MIXED
    }

    private enum EnemyCategory {
        CLOSE_RANGE_AGGRESSOR,
        PASSIVE_SURVIVOR,
        SPINNER_WEAK_BOT,
        HIGH_ACCURACY_THREAT,
        BALANCED
    }

    private static class AdaptivePolicy {
        double centerAversion = 0.35;
        double disengageBias = 0.50;
        double antiRamBias = 0.40;

        void update(Collection<EnemyModel> liveEnemies) {
            int aggressors = 0;
            int passive = 0;
            int spinners = 0;
            int snipers = 0;

            for (EnemyModel enemy : liveEnemies) {
                switch (enemy.category) {
                    case CLOSE_RANGE_AGGRESSOR:
                        aggressors++;
                        break;
                    case PASSIVE_SURVIVOR:
                        passive++;
                        break;
                    case SPINNER_WEAK_BOT:
                        spinners++;
                        break;
                    case HIGH_ACCURACY_THREAT:
                        snipers++;
                        break;
                    default:
                        break;
                }
            }

            centerAversion = 0.30 + aggressors * 0.08 + snipers * 0.05;
            disengageBias = 0.45 + aggressors * 0.10;
            antiRamBias = 0.35 + spinners * 0.06 + aggressors * 0.05;

            if (passive > aggressors + snipers) {
                centerAversion -= 0.08;
            }
        }
    }

    private static class EnemyModel {
        final String name;

        boolean alive = true;
        double energy = 100.0;
        double x;
        double y;
        double distance = 1000.0;
        double absoluteBearing;
        double headingRadians;
        double velocity;
        double lateralVelocity;
        double closingVelocity;
        long lastSeenTime = -1;
        int nearbyEnemyCount = 0;

        double aggressionLevel = 0.0;
        double averageDistance = 450.0;
        double firingFrequency = 0.0;
        double targetMeLikelihood = 0.0;
        double estimatedAccuracy = 0.0;
        double threatScore = 0.0;

        double headingChangeEma = 0.0;
        double velocityChangeEma = 0.0;
        double stationaryRatio = 0.0;
        double strafeRatio = 0.0;

        double lastHeadingRadians = 0.0;
        double lastVelocity = 0.0;
        double lastEnergy = 100.0;
        long scans = 0;
        long estimatedShots = 0;
        long closeContactCount = 0;
        long bulletsHitMe = 0;
        long bulletsIHit = 0;
        long bulletsIFired = 0;

        MovementStyle movementStyle = MovementStyle.MIXED;
        EnemyCategory category = EnemyCategory.BALANCED;

        EnemyModel(String name) {
            this.name = name;
        }

        void updateFromScan(AdvancedRobot robot, ScannedRobotEvent event, Point2D.Double myLocation) {
            alive = true;
            scans++;
            lastSeenTime = robot.getTime();

            absoluteBearing = robot.getHeadingRadians() + event.getBearingRadians();
            headingRadians = event.getHeadingRadians();
            velocity = event.getVelocity();
            distance = event.getDistance();
            energy = event.getEnergy();
            x = myLocation.x + Math.sin(absoluteBearing) * distance;
            y = myLocation.y + Math.cos(absoluteBearing) * distance;

            lateralVelocity = event.getVelocity() * Math.sin(event.getHeadingRadians() - absoluteBearing);
            closingVelocity = -event.getVelocity() * Math.cos(event.getHeadingRadians() - absoluteBearing);

            averageDistance = blend(averageDistance, distance, 0.08);
            stationaryRatio = blend(stationaryRatio, Math.abs(velocity) < 1.0 ? 1.0 : 0.0, 0.10);
            strafeRatio = blend(strafeRatio, Math.abs(lateralVelocity) > 4.0 ? 1.0 : 0.0, 0.10);
            headingChangeEma = blend(headingChangeEma, Math.abs(Utils.normalRelativeAngle(headingRadians - lastHeadingRadians)), 0.18);
            velocityChangeEma = blend(velocityChangeEma, Math.abs(velocity - lastVelocity), 0.18);

            double energyDrop = lastEnergy - energy;
            if (energyDrop >= 0.1 && energyDrop <= 3.0) {
                estimatedShots++;
                firingFrequency = blend(firingFrequency, 1.0, 0.18);
                if (distance < 280.0 || Math.abs(event.getBearingRadians()) < Math.toRadians(20.0)) {
                    targetMeLikelihood = blend(targetMeLikelihood, 1.0, 0.14);
                }
            } else {
                firingFrequency = blend(firingFrequency, 0.0, 0.04);
            }

            double closePressure = clamp(1.0 - distance / 250.0, 0.0, 1.0);
            double chasePressure = clamp(closingVelocity / 8.0, 0.0, 1.0);
            double ramPressure = clamp(closeContactCount / 3.0, 0.0, 1.0);
            aggressionLevel = clamp(
                0.45 * blend(aggressionLevel, closePressure, 0.10) +
                0.35 * chasePressure +
                0.20 * ramPressure,
                0.0,
                1.0
            );

            updateMovementStyle();
            refreshDerivedMetrics();

            lastHeadingRadians = headingRadians;
            lastVelocity = velocity;
            lastEnergy = energy;
        }

        void refreshDerivedMetrics() {
            if (estimatedShots > 0) {
                estimatedAccuracy = clamp((double) bulletsHitMe / (double) estimatedShots, 0.0, 1.0);
            }

            category = classify();

            double proximity = clamp(1.0 - distance / 600.0, 0.0, 1.0);
            double energyFactor = clamp(energy / 100.0, 0.0, 1.0);
            double categoryBonus = 0.0;
            if (category == EnemyCategory.CLOSE_RANGE_AGGRESSOR) {
                categoryBonus += 0.12;
            } else if (category == EnemyCategory.HIGH_ACCURACY_THREAT) {
                categoryBonus += 0.15;
            } else if (category == EnemyCategory.SPINNER_WEAK_BOT) {
                categoryBonus -= 0.10;
            } else if (category == EnemyCategory.PASSIVE_SURVIVOR) {
                categoryBonus -= 0.04;
            }

            threatScore = clamp(
                0.27 * aggressionLevel +
                0.18 * proximity +
                0.18 * firingFrequency +
                0.17 * targetMeLikelihood +
                0.12 * estimatedAccuracy +
                0.08 * energyFactor +
                categoryBonus,
                0.0,
                1.0
            );
        }

        void registerHitOnMe() {
            bulletsHitMe++;
            targetMeLikelihood = blend(targetMeLikelihood, 1.0, 0.22);
        }

        void registerMyShot() {
            bulletsIFired++;
        }

        void registerMyHit() {
            bulletsIHit++;
        }

        private void updateMovementStyle() {
            if (headingChangeEma > 0.45 && velocityChangeEma < 1.2) {
                movementStyle = MovementStyle.SPINNER;
            } else if (strafeRatio > 0.55) {
                movementStyle = MovementStyle.STRAFER;
            } else if (aggressionLevel > 0.72 && averageDistance < 180.0) {
                movementStyle = MovementStyle.RAMMER;
            } else if (stationaryRatio > 0.45) {
                movementStyle = MovementStyle.CAMPER;
            } else {
                movementStyle = MovementStyle.MIXED;
            }
        }

        private EnemyCategory classify() {
            if (movementStyle == MovementStyle.SPINNER || stationaryRatio > 0.58) {
                return EnemyCategory.SPINNER_WEAK_BOT;
            }
            if (estimatedAccuracy > 0.18 && firingFrequency > 0.12) {
                return EnemyCategory.HIGH_ACCURACY_THREAT;
            }
            if (aggressionLevel > 0.60 && averageDistance < 260.0) {
                return EnemyCategory.CLOSE_RANGE_AGGRESSOR;
            }
            if (aggressionLevel < 0.28 && firingFrequency < 0.06 && averageDistance > 320.0) {
                return EnemyCategory.PASSIVE_SURVIVOR;
            }
            return EnemyCategory.BALANCED;
        }

        private static double blend(double current, double observation, double weight) {
            return current + (observation - current) * weight;
        }
    }
}
