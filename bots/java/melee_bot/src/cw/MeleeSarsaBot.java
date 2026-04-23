package cw;

import robocode.AdvancedRobot;
import robocode.BattleEndedEvent;
import robocode.BulletHitEvent;
import robocode.DeathEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;
import robocode.WinEvent;
import robocode.util.Utils;

import java.awt.Color;
import java.awt.geom.Point2D;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

public class MeleeSarsaBot extends AdvancedRobot {
    private static final int ORBIT_CLOCKWISE = 0;
    private static final int ORBIT_COUNTERCLOCKWISE = 1;
    private static final int RETREAT_CLUSTER = 2;
    private static final int ADVANCE_OPEN_SPACE = 3;
    private static final int RADAR_SWEEP_LEFT = 4;
    private static final int RADAR_SWEEP_RIGHT = 5;
    private static final int FIRE_LOW = 6;
    private static final int FIRE_MEDIUM = 7;
    private static final int FIRE_HIGH = 8;
    private static final int EVADE = 9;
    private static final int ACTION_COUNT = 10;

    private static final double ALPHA = 0.12;
    private static final double GAMMA = 0.95;
    private static final double EPSILON_MIN = 0.08;
    private static final double EPSILON_DECAY = 0.9975;
    private static final double STALE_SCAN_TICKS = 24.0;
    private static final double WALL_MARGIN = 72.0;
    private static final SarsaTable SHARED_Q_TABLE = new SarsaTable(ACTION_COUNT);

    private static boolean qTableLoaded;
    private static double sharedEpsilon = 1.0;
    private static int sharedEpisodeCounter;

    private final Random random = new Random();
    private final Map<String, EnemySnapshot> enemies = new HashMap<>();

    private String currentTargetName;
    private String previousState;
    private Integer previousAction;

    private double epsilon = 1.0;
    private double stepRewardAccumulator;
    private double episodeReward;
    private double tdAbsSum;
    private int tdCount;
    private int crowdedTicks;
    private int wallHits;
    private int fireActions;
    private int episodeNumber;
    private int startingOpponentCount;

    public void run() {
        if (!qTableLoaded) {
            SHARED_Q_TABLE.load(qTablePath());
            qTableLoaded = true;
        }
        epsilon = sharedEpsilon;

        setBodyColor(new Color(44, 75, 122));
        setGunColor(new Color(232, 193, 112));
        setRadarColor(new Color(247, 246, 243));
        setBulletColor(new Color(245, 145, 71));
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setAdjustRadarForRobotTurn(true);
        setMaxVelocity(8.0);

        resetRoundState();

        while (true) {
            startingOpponentCount = Math.max(startingOpponentCount, getOthers());
            removeStaleEnemies();

            StateView stateView = buildStateView();
            stepRewardAccumulator += 0.01;
            updateCrowdingPenalty(stateView);

            int action = selectAction(stateView);
            if (previousState != null && previousAction != null) {
                sarsaUpdate(previousState, previousAction, stepRewardAccumulator, stateView.key, action, false);
                episodeReward += stepRewardAccumulator;
            }

            stepRewardAccumulator = 0.0;
            executeAction(action, stateView);
            previousState = stateView.key;
            previousAction = action;

            execute();
        }
    }

    public void onScannedRobot(ScannedRobotEvent event) {
        double absBearing = getHeadingRadians() + event.getBearingRadians();
        double enemyX = getX() + Math.sin(absBearing) * event.getDistance();
        double enemyY = getY() + Math.cos(absBearing) * event.getDistance();

        EnemySnapshot snapshot = enemies.computeIfAbsent(event.getName(), EnemySnapshot::new);
        snapshot.x = enemyX;
        snapshot.y = enemyY;
        snapshot.distance = event.getDistance();
        snapshot.absBearingRadians = absBearing;
        snapshot.relBearingRadians = event.getBearingRadians();
        snapshot.energy = event.getEnergy();
        snapshot.velocity = event.getVelocity();
        snapshot.headingRadians = event.getHeadingRadians();
        snapshot.lateralVelocity = event.getVelocity()
            * Math.sin(event.getHeadingRadians() - absBearing);
        snapshot.lastSeenTick = getTime();
        snapshot.alive = true;

        chooseTarget(snapshot.name, true);
    }

    public void onRobotDeath(RobotDeathEvent event) {
        EnemySnapshot snapshot = enemies.get(event.getName());
        if (snapshot != null) {
            snapshot.alive = false;
            if (snapshot.age(getTime()) <= 2) {
                stepRewardAccumulator += 1.35;
            }
        }
        if (event.getName().equals(currentTargetName)) {
            currentTargetName = null;
        }
    }

    public void onBulletHit(BulletHitEvent event) {
        double power = event.getBullet() != null ? event.getBullet().getPower() : 1.0;
        double damage = bulletDamage(power);
        stepRewardAccumulator += 0.070 * damage;

        EnemySnapshot snapshot = enemies.get(event.getName());
        if (snapshot != null) {
            snapshot.energy = Math.max(0.0, snapshot.energy - damage);
        }
    }

    public void onHitByBullet(HitByBulletEvent event) {
        double power = event.getBullet() != null ? event.getBullet().getPower() : 1.0;
        double damage = bulletDamage(power);
        stepRewardAccumulator -= 0.095 * damage;
    }

    public void onHitWall(HitWallEvent event) {
        wallHits++;
        stepRewardAccumulator -= 0.60;
        setBack(90);
        setTurnRight(normalizedDegrees(120 - event.getBearing()));
    }

    public void onHitRobot(HitRobotEvent event) {
        double penalty = 0.25;
        if (computeDangerBucket() >= 2) {
            penalty += 0.35;
        }
        stepRewardAccumulator -= penalty;
        if (event.isMyFault()) {
            setBack(60);
        }
    }

    public void onWin(WinEvent event) {
        finishEpisode(true);
    }

    public void onDeath(DeathEvent event) {
        finishEpisode(false);
    }

    public void onBattleEnded(BattleEndedEvent event) {
        SHARED_Q_TABLE.save(qTablePath());
    }

    private void resetRoundState() {
        episodeNumber = ++sharedEpisodeCounter;
        enemies.clear();
        currentTargetName = null;
        previousState = null;
        previousAction = null;
        stepRewardAccumulator = 0.0;
        episodeReward = 0.0;
        tdAbsSum = 0.0;
        tdCount = 0;
        crowdedTicks = 0;
        wallHits = 0;
        fireActions = 0;
        startingOpponentCount = getOthers();
    }

    private StateView buildStateView() {
        chooseTarget(currentTargetName, true);

        EnemySnapshot nearest = freshestNearestEnemy();
        EnemySnapshot weakest = freshestWeakestEnemy();
        EnemySnapshot target = currentTarget();

        int myEnergy = bucketMyEnergy(getEnergy());
        int nearestDistance = bucketDistance(nearest != null ? nearest.distance : 900.0);
        int nearestBearing = bucketBearing(nearest != null ? nearest.relBearingRadians : 0.0);
        int weakestDistance = bucketCompactDistance(weakest != null ? weakest.distance : 900.0);
        int nearbyEnemies = bucketNearbyEnemies(countNearbyEnemies(260.0));
        int danger = computeDangerBucket();
        int wall = bucketWall(minWallDistance());
        int gunReady = getGunHeat() <= 0.0001 ? 1 : 0;
        int targetEnergy = bucketTargetEnergy(target != null ? target.energy : 100.0);
        int targetDistance = bucketDistance(target != null ? target.distance : 900.0);

        String key = "me" + myEnergy
            + "|nd" + nearestDistance
            + "|nb" + nearestBearing
            + "|wd" + weakestDistance
            + "|nn" + nearbyEnemies
            + "|dg" + danger
            + "|wp" + wall
            + "|gr" + gunReady
            + "|te" + targetEnergy
            + "|td" + targetDistance;

        return new StateView(key, danger, wall, gunReady == 1, target, nearest);
    }

    private void chooseTarget(String preferredName, boolean allowStickyHold) {
        EnemySnapshot incumbent = preferredName != null ? enemies.get(preferredName) : null;
        if (incumbent != null && (!incumbent.alive || incumbent.age(getTime()) > STALE_SCAN_TICKS)) {
            incumbent = null;
        }

        EnemySnapshot best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (EnemySnapshot enemy : enemies.values()) {
            if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
                continue;
            }

            double distanceScore = 1.35 * (1.0 - clamp(enemy.distance / 800.0, 0.0, 1.0));
            double weaknessScore = 1.10 * (1.0 - clamp(enemy.energy / 100.0, 0.0, 1.0));
            double freshnessScore = 1.00 * (1.0 - clamp(enemy.age(getTime()) / STALE_SCAN_TICKS, 0.0, 1.0));
            double aimEaseScore = 0.90 * (1.0 - clamp(Math.abs(enemy.lateralVelocity) / 8.0, 0.0, 1.0));
            double bearingScore = 0.45 * (1.0 - clamp(Math.abs(enemy.relBearingRadians) / Math.PI, 0.0, 1.0));
            double stickiness = enemy.name.equals(currentTargetName) ? 0.70 : 0.0;

            double score = distanceScore + weaknessScore + freshnessScore + aimEaseScore + bearingScore + stickiness;
            if (score > bestScore) {
                bestScore = score;
                best = enemy;
            }
        }

        if (best == null) {
            currentTargetName = null;
            return;
        }

        if (allowStickyHold && incumbent != null) {
            double incumbentScore = scoreEnemy(incumbent);
            if (incumbentScore >= bestScore - 0.35) {
                currentTargetName = incumbent.name;
                return;
            }
        }

        currentTargetName = best.name;
    }

    private double scoreEnemy(EnemySnapshot enemy) {
        if (enemy == null || !enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
            return Double.NEGATIVE_INFINITY;
        }

        double distanceScore = 1.35 * (1.0 - clamp(enemy.distance / 800.0, 0.0, 1.0));
        double weaknessScore = 1.10 * (1.0 - clamp(enemy.energy / 100.0, 0.0, 1.0));
        double freshnessScore = 1.00 * (1.0 - clamp(enemy.age(getTime()) / STALE_SCAN_TICKS, 0.0, 1.0));
        double aimEaseScore = 0.90 * (1.0 - clamp(Math.abs(enemy.lateralVelocity) / 8.0, 0.0, 1.0));
        double bearingScore = 0.45 * (1.0 - clamp(Math.abs(enemy.relBearingRadians) / Math.PI, 0.0, 1.0));
        double stickiness = enemy.name.equals(currentTargetName) ? 0.70 : 0.0;
        return distanceScore + weaknessScore + freshnessScore + aimEaseScore + bearingScore + stickiness;
    }

    private int selectAction(StateView stateView) {
        if (random.nextDouble() < epsilon) {
            return exploreAction(stateView);
        }

        double[] values = SHARED_Q_TABLE.get(stateView.key);
        double best = Double.NEGATIVE_INFINITY;
        int bestAction = 0;
        int ties = 0;

        for (int i = 0; i < values.length; i++) {
            double adjusted = values[i] + actionBias(i, stateView);
            if (adjusted > best + 1e-9) {
                best = adjusted;
                bestAction = i;
                ties = 1;
            } else if (Math.abs(adjusted - best) <= 1e-9 && random.nextInt(++ties) == 0) {
                bestAction = i;
            }
        }

        return bestAction;
    }

    private int exploreAction(StateView stateView) {
        double[] weights = new double[ACTION_COUNT];
        for (int i = 0; i < ACTION_COUNT; i++) {
            weights[i] = 1.0 + actionBias(i, stateView);
            if (weights[i] < 0.15) {
                weights[i] = 0.15;
            }
        }

        double total = 0.0;
        for (double weight : weights) {
            total += weight;
        }

        double draw = random.nextDouble() * total;
        for (int i = 0; i < weights.length; i++) {
            draw -= weights[i];
            if (draw <= 0.0) {
                return i;
            }
        }
        return ACTION_COUNT - 1;
    }

    private double actionBias(int action, StateView stateView) {
        double bias = 0.0;
        boolean targetMissing = stateView.target == null || stateView.target.age(getTime()) > 6;
        boolean gunAligned = stateView.target != null
            && Math.abs(normalizedDegrees(Math.toDegrees(
                absoluteBearing(getX(), getY(), stateView.target.x, stateView.target.y) - getGunHeadingRadians()
            ))) < 11.0;

        if (targetMissing && (action == RADAR_SWEEP_LEFT || action == RADAR_SWEEP_RIGHT)) {
            bias += 2.7;
        }

        if (stateView.dangerBucket >= 2) {
            if (action == EVADE) {
                bias += 3.0;
            }
            if (action == RETREAT_CLUSTER) {
                bias += 2.4;
            }
            if (action == ORBIT_CLOCKWISE || action == ORBIT_COUNTERCLOCKWISE) {
                bias += 1.5;
            }
            if (action == FIRE_HIGH) {
                bias -= 0.7;
            }
        } else if (stateView.target != null) {
            if (action == ORBIT_CLOCKWISE || action == ORBIT_COUNTERCLOCKWISE) {
                bias += 1.2;
            }
            if (action == ADVANCE_OPEN_SPACE) {
                bias += 0.8;
            }
        }

        if (!stateView.gunReady || !gunAligned) {
            if (action == FIRE_LOW || action == FIRE_MEDIUM || action == FIRE_HIGH) {
                bias -= 0.9;
            }
        } else {
            if (action == FIRE_LOW || action == FIRE_MEDIUM) {
                bias += 0.9;
            }
            if (action == FIRE_HIGH && stateView.dangerBucket <= 1) {
                bias += 0.5;
            }
        }

        if (stateView.wallBucket >= 2) {
            if (action == ADVANCE_OPEN_SPACE || action == EVADE || action == RETREAT_CLUSTER) {
                bias += 1.8;
            }
            if (action == ORBIT_CLOCKWISE || action == ORBIT_COUNTERCLOCKWISE) {
                bias += 0.5;
            }
        }

        return bias;
    }

    private void executeAction(int action, StateView stateView) {
        EnemySnapshot target = stateView.target;
        switch (action) {
            case ORBIT_CLOCKWISE:
                orbitTarget(target, -1);
                break;
            case ORBIT_COUNTERCLOCKWISE:
                orbitTarget(target, 1);
                break;
            case RETREAT_CLUSTER:
                retreatFromCluster();
                break;
            case ADVANCE_OPEN_SPACE:
                advanceIntoOpenSpace();
                break;
            case RADAR_SWEEP_LEFT:
                setTurnRadarLeft(85);
                trackGun(target);
                break;
            case RADAR_SWEEP_RIGHT:
                setTurnRadarRight(85);
                trackGun(target);
                break;
            case FIRE_LOW:
                fireIfAligned(target, 1.0);
                break;
            case FIRE_MEDIUM:
                fireIfAligned(target, 1.8);
                break;
            case FIRE_HIGH:
                fireIfAligned(target, 2.6);
                break;
            case EVADE:
                evade();
                break;
            default:
                advanceIntoOpenSpace();
                break;
        }

        maintainRadarLock(target);
    }

    private void orbitTarget(EnemySnapshot target, int orbitDirection) {
        if (target == null) {
            advanceIntoOpenSpace();
            return;
        }

        double absBearing = absoluteBearing(getX(), getY(), target.x, target.y);
        double desired = absBearing + orbitDirection * Math.PI / 2.0;
        if (target.distance < 150.0) {
            desired += orbitDirection * Math.PI / 10.0;
        }
        desired = wallSmoothedHeading(desired, 110.0, orbitDirection);
        setBackAsFront(desired, 120.0);
        trackGun(target);
    }

    private void retreatFromCluster() {
        Point2D.Double clusterCenter = clusterCenter(320.0);
        double heading;
        if (clusterCenter != null) {
            heading = absoluteBearing(clusterCenter.x, clusterCenter.y, getX(), getY());
        } else {
            EnemySnapshot nearest = freshestNearestEnemy();
            if (nearest == null) {
                heading = bestOpenSpaceHeading();
            } else {
                heading = absoluteBearing(nearest.x, nearest.y, getX(), getY());
            }
        }

        heading = wallSmoothedHeading(heading, 150.0, 1);
        setBackAsFront(heading, 160.0);
        trackGun(currentTarget());
    }

    private void advanceIntoOpenSpace() {
        double heading = wallSmoothedHeading(bestOpenSpaceHeading(), 130.0, 1);
        setBackAsFront(heading, 130.0);
        trackGun(currentTarget());
    }

    private void evade() {
        double heading = wallSmoothedHeading(bestOpenSpaceHeading(), 180.0, random.nextBoolean() ? 1 : -1);
        setMaxVelocity(8.0);
        setBackAsFront(heading, 180.0);
        EnemySnapshot target = currentTarget();
        if (target != null) {
            setTurnGunRightRadians(Utils.normalRelativeAngle(target.absBearingRadians - getGunHeadingRadians()));
        }
    }

    private void fireIfAligned(EnemySnapshot target, double requestedPower) {
        fireActions++;
        if (target == null) {
            setTurnRadarRight(90);
            return;
        }

        double power = requestedPower;
        if (getEnergy() < 22.0) {
            power = Math.min(power, 1.4);
        }
        if (computeDangerBucket() >= 2) {
            power = Math.min(power, 1.8);
        }

        double gunTurn = predictiveGunTurn(target, power);
        setTurnGunRightRadians(gunTurn);
        double absTurn = Math.abs(Math.toDegrees(gunTurn));
        if (getGunHeat() <= 0.0001 && getEnergy() > power + 0.15 && absTurn < 8.0) {
            setFire(power);
            stepRewardAccumulator -= 0.015;
        }
    }

    private void maintainRadarLock(EnemySnapshot target) {
        if (target != null && target.age(getTime()) <= 4) {
            double radarTurn = Utils.normalRelativeAngle(target.absBearingRadians - getRadarHeadingRadians());
            setTurnRadarRightRadians(radarTurn * 2.0);
        } else {
            setTurnRadarRight(60);
        }
    }

    private void trackGun(EnemySnapshot target) {
        if (target == null) {
            return;
        }
        setTurnGunRightRadians(predictiveGunTurn(target, 1.8));
    }

    private double predictiveGunTurn(EnemySnapshot target, double power) {
        double bulletSpeed = 20.0 - 3.0 * power;
        double fireTime = target.distance / Math.max(11.0, bulletSpeed);
        double predictedX = clamp(target.x + Math.sin(target.headingRadians) * target.velocity * fireTime, WALL_MARGIN,
            getBattleFieldWidth() - WALL_MARGIN);
        double predictedY = clamp(target.y + Math.cos(target.headingRadians) * target.velocity * fireTime, WALL_MARGIN,
            getBattleFieldHeight() - WALL_MARGIN);
        double aimBearing = absoluteBearing(getX(), getY(), predictedX, predictedY);
        return Utils.normalRelativeAngle(aimBearing - getGunHeadingRadians());
    }

    private void setBackAsFront(double goAngle, double distance) {
        double angle = Utils.normalRelativeAngle(goAngle - getHeadingRadians());
        if (Math.abs(angle) > Math.PI / 2.0) {
            if (angle < 0.0) {
                setTurnRightRadians(Math.PI + angle);
            } else {
                setTurnLeftRadians(Math.PI - angle);
            }
            setBack(distance);
        } else {
            if (angle < 0.0) {
                setTurnLeftRadians(-angle);
            } else {
                setTurnRightRadians(angle);
            }
            setAhead(distance);
        }
    }

    private double wallSmoothedHeading(double angle, double distance, int turnDirection) {
        double smoothed = angle;
        for (int i = 0; i < 24; i++) {
            double testX = getX() + Math.sin(smoothed) * distance;
            double testY = getY() + Math.cos(smoothed) * distance;
            if (testX > WALL_MARGIN && testX < getBattleFieldWidth() - WALL_MARGIN
                && testY > WALL_MARGIN && testY < getBattleFieldHeight() - WALL_MARGIN) {
                break;
            }
            smoothed += turnDirection * 0.18;
        }
        return smoothed;
    }

    private double bestOpenSpaceHeading() {
        double bestScore = Double.NEGATIVE_INFINITY;
        double bestHeading = getHeadingRadians();

        for (int i = 0; i < 16; i++) {
            double heading = i * (Math.PI / 8.0);
            double testX = clamp(getX() + Math.sin(heading) * 170.0, 18.0, getBattleFieldWidth() - 18.0);
            double testY = clamp(getY() + Math.cos(heading) * 170.0, 18.0, getBattleFieldHeight() - 18.0);
            double wallScore = Math.min(minWallDistance(testX, testY), 180.0) / 180.0 * 2.4;
            double enemyScore = 0.0;

            for (EnemySnapshot enemy : enemies.values()) {
                if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
                    continue;
                }
                double distance = Point2D.distance(testX, testY, enemy.x, enemy.y);
                enemyScore += Math.min(distance, 500.0) / 500.0;
                if (distance < 180.0) {
                    enemyScore -= 1.2;
                }
            }

            double score = wallScore + enemyScore;
            if (score > bestScore) {
                bestScore = score;
                bestHeading = heading;
            }
        }

        return bestHeading;
    }

    private Point2D.Double clusterCenter(double maxDistance) {
        double sumX = 0.0;
        double sumY = 0.0;
        int count = 0;

        for (EnemySnapshot enemy : enemies.values()) {
            if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS || enemy.distance > maxDistance) {
                continue;
            }
            sumX += enemy.x;
            sumY += enemy.y;
            count++;
        }

        if (count == 0) {
            return null;
        }
        return new Point2D.Double(sumX / count, sumY / count);
    }

    private EnemySnapshot freshestNearestEnemy() {
        EnemySnapshot nearest = null;
        double bestDistance = Double.POSITIVE_INFINITY;

        for (EnemySnapshot enemy : enemies.values()) {
            if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
                continue;
            }
            if (enemy.distance < bestDistance) {
                bestDistance = enemy.distance;
                nearest = enemy;
            }
        }

        return nearest;
    }

    private EnemySnapshot freshestWeakestEnemy() {
        EnemySnapshot weakest = null;
        double bestEnergy = Double.POSITIVE_INFINITY;
        double bestDistance = Double.POSITIVE_INFINITY;

        for (EnemySnapshot enemy : enemies.values()) {
            if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
                continue;
            }
            if (enemy.energy < bestEnergy - 1e-9
                || (Math.abs(enemy.energy - bestEnergy) <= 1e-9 && enemy.distance < bestDistance)) {
                bestEnergy = enemy.energy;
                bestDistance = enemy.distance;
                weakest = enemy;
            }
        }

        return weakest;
    }

    private EnemySnapshot currentTarget() {
        if (currentTargetName == null) {
            return null;
        }
        EnemySnapshot target = enemies.get(currentTargetName);
        if (target == null || !target.alive || target.age(getTime()) > STALE_SCAN_TICKS) {
            currentTargetName = null;
            return null;
        }
        return target;
    }

    private int computeDangerBucket() {
        double danger = 0.0;
        for (EnemySnapshot enemy : enemies.values()) {
            if (!enemy.alive || enemy.age(getTime()) > STALE_SCAN_TICKS) {
                continue;
            }
            double distanceFactor = Math.max(0.0, 320.0 - enemy.distance) / 320.0;
            double energyFactor = 0.5 + clamp(enemy.energy / 100.0, 0.0, 1.0);
            danger += distanceFactor * energyFactor;
        }

        if (minWallDistance() < 85.0) {
            danger += 0.7;
        }
        if (getEnergy() < 25.0) {
            danger += 0.6;
        }

        if (danger < 0.7) {
            return 0;
        }
        if (danger < 1.5) {
            return 1;
        }
        if (danger < 2.5) {
            return 2;
        }
        return 3;
    }

    private void updateCrowdingPenalty(StateView stateView) {
        if (stateView.dangerBucket >= 2 && countNearbyEnemies(220.0) >= 2) {
            crowdedTicks++;
            if (crowdedTicks > 5) {
                stepRewardAccumulator -= 0.040;
            }
        } else {
            crowdedTicks = 0;
        }
    }

    private int countNearbyEnemies(double radius) {
        int count = 0;
        for (EnemySnapshot enemy : enemies.values()) {
            if (enemy.alive && enemy.age(getTime()) <= STALE_SCAN_TICKS && enemy.distance <= radius) {
                count++;
            }
        }
        return count;
    }

    private void removeStaleEnemies() {
        for (EnemySnapshot enemy : enemies.values()) {
            if (enemy.age(getTime()) > 80) {
                enemy.alive = false;
            }
        }
    }

    private void sarsaUpdate(String state, int action, double reward, String nextState, int nextAction, boolean done) {
        double[] currentValues = SHARED_Q_TABLE.get(state);
        double[] nextValues = SHARED_Q_TABLE.get(nextState);
        double target = reward;
        if (!done) {
            target += GAMMA * nextValues[nextAction];
        }

        double tdError = target - currentValues[action];
        currentValues[action] += ALPHA * tdError;
        tdAbsSum += Math.abs(tdError);
        tdCount++;
    }

    private void finishEpisode(boolean won) {
        int placement = won ? 1 : getOthers() + 1;
        int totalBots = Math.max(2, startingOpponentCount + 1);
        double placementBonus = 1.8 * (double) (totalBots - placement) / (double) (totalBots - 1);
        double terminalReward = won ? 2.4 : -1.1;

        stepRewardAccumulator += terminalReward + placementBonus;

        if (previousState != null && previousAction != null) {
            double[] values = SHARED_Q_TABLE.get(previousState);
            double tdError = stepRewardAccumulator - values[previousAction];
            values[previousAction] += ALPHA * tdError;
            tdAbsSum += Math.abs(tdError);
            tdCount++;
            episodeReward += stepRewardAccumulator;
        }

        double avgAbsTd = tdCount == 0 ? 0.0 : tdAbsSum / tdCount;
        appendLog(won, placement, avgAbsTd);
        SHARED_Q_TABLE.save(qTablePath());
        epsilon = Math.max(EPSILON_MIN, epsilon * EPSILON_DECAY);
        sharedEpsilon = epsilon;
    }

    private void appendLog(boolean won, int placement, double avgAbsTd) {
        String row = String.format(
            Locale.US,
            "{\"episode\":%d,\"won\":%s,\"placement\":%d,\"epsilon\":%.5f,"
                + "\"total_reward\":%.4f,\"avg_abs_td_error\":%.6f,"
                + "\"wall_hits\":%d,\"fire_actions\":%d,\"turns\":%d}%n",
            episodeNumber,
            won ? "true" : "false",
            placement,
            epsilon,
            episodeReward,
            avgAbsTd,
            wallHits,
            fireActions,
            getTime()
        );

        try {
            Path logPath = logPath();
            Path parent = logPath.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }
            Files.write(
                logPath,
                row.getBytes(StandardCharsets.UTF_8),
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
            );
        } catch (IOException ignored) {
            // Ignore logging failures mid-battle.
        }
    }

    private static double bulletDamage(double power) {
        return 4.0 * power + Math.max(0.0, 2.0 * (power - 1.0));
    }

    private int bucketMyEnergy(double energy) {
        if (energy < 18.0) {
            return 0;
        }
        if (energy < 38.0) {
            return 1;
        }
        if (energy < 68.0) {
            return 2;
        }
        return 3;
    }

    private int bucketTargetEnergy(double energy) {
        if (energy < 12.0) {
            return 0;
        }
        if (energy < 30.0) {
            return 1;
        }
        if (energy < 60.0) {
            return 2;
        }
        return 3;
    }

    private int bucketDistance(double distance) {
        if (distance < 130.0) {
            return 0;
        }
        if (distance < 280.0) {
            return 1;
        }
        if (distance < 480.0) {
            return 2;
        }
        return 3;
    }

    private int bucketCompactDistance(double distance) {
        if (distance < 170.0) {
            return 0;
        }
        if (distance < 420.0) {
            return 1;
        }
        return 2;
    }

    private int bucketBearing(double bearingRadians) {
        double degrees = Math.toDegrees(Utils.normalRelativeAngle(bearingRadians));
        if (degrees < -60.0) {
            return 0;
        }
        if (degrees < 0.0) {
            return 1;
        }
        if (degrees < 60.0) {
            return 2;
        }
        return 3;
    }

    private int bucketNearbyEnemies(int nearbyEnemies) {
        if (nearbyEnemies <= 0) {
            return 0;
        }
        if (nearbyEnemies == 1) {
            return 1;
        }
        if (nearbyEnemies == 2) {
            return 2;
        }
        return 3;
    }

    private int bucketWall(double wallDistance) {
        if (wallDistance < 60.0) {
            return 2;
        }
        if (wallDistance < 140.0) {
            return 1;
        }
        return 0;
    }

    private double minWallDistance() {
        return minWallDistance(getX(), getY());
    }

    private double minWallDistance(double x, double y) {
        return Math.min(
            Math.min(x, getBattleFieldWidth() - x),
            Math.min(y, getBattleFieldHeight() - y)
        );
    }

    private double absoluteBearing(double sourceX, double sourceY, double targetX, double targetY) {
        return Math.atan2(targetX - sourceX, targetY - sourceY);
    }

    private Path qTablePath() {
        return botDataPath("melee_sarsa_qtable.tsv");
    }

    private Path logPath() {
        return botDataPath("melee_sarsa_log.jsonl");
    }

    private Path botDataPath(String fileName) {
        File file = getDataFile(fileName);
        if (file != null) {
            return file.toPath();
        }
        return Paths.get(fileName);
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private double normalizedDegrees(double degrees) {
        return Utils.normalRelativeAngleDegrees(degrees);
    }

    private static final class StateView {
        final String key;
        final int dangerBucket;
        final int wallBucket;
        final boolean gunReady;
        final EnemySnapshot target;
        final EnemySnapshot nearest;

        StateView(String key, int dangerBucket, int wallBucket, boolean gunReady,
                  EnemySnapshot target, EnemySnapshot nearest) {
            this.key = key;
            this.dangerBucket = dangerBucket;
            this.wallBucket = wallBucket;
            this.gunReady = gunReady;
            this.target = target;
            this.nearest = nearest;
        }
    }
}
