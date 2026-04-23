package cw.meleedqn;

import robocode.AdvancedRobot;
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
import java.io.IOException;

public class MeleeDqnBot extends AdvancedRobot {
    private final EnemyManager enemyManager = new EnemyManager();
    private final TargetSelector targetSelector = new TargetSelector();
    private final DangerMap dangerMap = new DangerMap();
    private final StateEncoder stateEncoder = new StateEncoder();
    private final RewardTracker rewardTracker = new RewardTracker();
    private final SocketDqnClient dqnClient = new SocketDqnClient("localhost", 5000);

    private int episode;
    private int initialBotCount;
    private long localTick;
    private long lastDamageTick;
    private int lastAppliedSwitchCount;
    private ActionType lastAction = ActionType.HEAD_TO_OPEN_SPACE;
    private String lastDamagedEnemyName;

    @Override
    public void run() {
        episode++;
        initialBotCount = getOthers() + 1;
        localTick = 0L;
        lastDamageTick = -999L;
        lastAppliedSwitchCount = 0;
        lastAction = ActionType.HEAD_TO_OPEN_SPACE;
        lastDamagedEnemyName = null;

        enemyManager.resetRound();
        targetSelector.resetRound();
        rewardTracker.resetRound();

        setAdjustRadarForGunTurn(true);
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForRobotTurn(true);
        setMaxVelocity(8.0);
        setColors(new Color(22, 45, 80), new Color(247, 188, 48), new Color(219, 76, 76));

        try {
            dqnClient.connectIfNeeded();
        } catch (IOException e) {
            out.println("DQN socket unavailable, falling back to open-space movement.");
        }

        setTurnRadarRightRadians(Double.POSITIVE_INFINITY);

        while (true) {
            localTick++;

            EnemySnapshot target = targetSelector.select(this, enemyManager.all(), localTick);
            int switchDelta = Math.max(0, targetSelector.getSwitchCount() - lastAppliedSwitchCount);
            lastAppliedSwitchCount = targetSelector.getSwitchCount();
            rewardTracker.onTick(this, dangerMap.crowdingScore(this, enemyManager.all()), switchDelta);

            double[] state = stateEncoder.encode(this, enemyManager, targetSelector, dangerMap, rewardTracker, localTick);
            int actionId = requestAction(state);
            ActionType action = sanitizeAction(ActionType.fromId(actionId), target);

            aimGunAt(target);
            executeAction(action, target);
            updateRadar(target);
            execute();
            lastAction = action;
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        enemyManager.update(this, event, getTime());
    }

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        enemyManager.onRobotDeath(event);
        if (event.getName().equals(lastDamagedEnemyName) && (getTime() - lastDamageTick) <= 2L) {
            rewardTracker.onKill();
        }
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        lastDamagedEnemyName = event.getName();
        lastDamageTick = getTime();
        rewardTracker.onBulletDamageDealt(bulletDamage(event.getBullet().getPower()));
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        rewardTracker.onBulletDamageTaken(bulletDamage(event.getBullet().getPower()));
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        rewardTracker.onHitWall();
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        rewardTracker.onRobotCollision();
    }

    @Override
    public void onDeath(DeathEvent event) {
        sendTerminal(false);
    }

    @Override
    public void onWin(WinEvent event) {
        sendTerminal(true);
    }

    private int requestAction(double[] state) {
        BattleStats stats = buildStats(0);
        try {
            return dqnClient.requestAction(state, rewardTracker.consumeStepReward(), false, stats);
        } catch (IOException e) {
            return ActionType.HEAD_TO_OPEN_SPACE.ordinal();
        }
    }

    private BattleStats buildStats(int placement) {
        BattleStats stats = new BattleStats();
        stats.episode = episode;
        stats.tick = localTick;
        stats.livingEnemies = getOthers();
        stats.placement = placement;
        stats.survivalTicks = rewardTracker.getSurvivalTicks();
        stats.damageDealt = rewardTracker.getDamageDealt();
        stats.damageTaken = rewardTracker.getDamageTaken();
        stats.kills = rewardTracker.getKills();
        stats.targetSwitches = targetSelector.getSwitchCount();
        return stats;
    }

    private void sendTerminal(boolean won) {
        int totalBots = Math.max(initialBotCount, getOthers() + 1);
        int placement = won ? 1 : getOthers() + 1;
        double terminalReward = rewardTracker.finishRound(placement, totalBots, won);

        double[] terminalState = new double[StateEncoder.STATE_SIZE];
        BattleStats stats = buildStats(placement);

        try {
            dqnClient.requestAction(terminalState, terminalReward, true, stats);
        } catch (IOException ignored) {
            // The trainer may have closed already; the match is over regardless.
        }
    }

    private void executeAction(ActionType action, EnemySnapshot target) {
        switch (action) {
            case AHEAD_SHORT:
                setAhead(80.0);
                break;
            case AHEAD_MEDIUM:
                setAhead(160.0);
                break;
            case BACK_SHORT:
                setBack(80.0);
                break;
            case BACK_MEDIUM:
                setBack(160.0);
                break;
            case TURN_LEFT_SMALL:
                setTurnLeft(15.0);
                setAhead(40.0);
                break;
            case TURN_RIGHT_SMALL:
                setTurnRight(15.0);
                setAhead(40.0);
                break;
            case TURN_LEFT_MEDIUM:
                setTurnLeft(35.0);
                setAhead(50.0);
                break;
            case TURN_RIGHT_MEDIUM:
                setTurnRight(35.0);
                setAhead(50.0);
                break;
            case STRAFE_LEFT:
                strafeTarget(target, -1.0);
                break;
            case STRAFE_RIGHT:
                strafeTarget(target, 1.0);
                break;
            case HEAD_TO_OPEN_SPACE:
                moveTowardHeading(dangerMap.safestHeading(this, enemyManager.all()), 140.0);
                break;
            case FLEE_CLUSTER:
                moveTowardHeading(dangerMap.escapeHeading(this, enemyManager.all()), 150.0);
                break;
            case FIRE_1:
                fireIfAligned(target, 1.0);
                break;
            case FIRE_2:
                fireIfAligned(target, 2.0);
                break;
            case FIRE_3:
                fireIfAligned(target, 3.0);
                break;
        }
    }

    private ActionType sanitizeAction(ActionType action, EnemySnapshot target) {
        if (action.isFireAction()) {
            if (target == null || getGunHeat() > 0.0 || Math.abs(target.gunTurnFrom(this)) > Math.toRadians(14.0)) {
                return dangerMap.crowdingScore(this, enemyManager.all()) > 0.45
                        ? ActionType.FLEE_CLUSTER
                        : ActionType.HEAD_TO_OPEN_SPACE;
            }
            return action;
        }

        if (dangerMap.forwardUnsafe(this) && (action == ActionType.AHEAD_SHORT || action == ActionType.AHEAD_MEDIUM)) {
            return ActionType.FLEE_CLUSTER;
        }

        if ((action == ActionType.STRAFE_LEFT || action == ActionType.STRAFE_RIGHT) && target == null) {
            return ActionType.HEAD_TO_OPEN_SPACE;
        }

        return action;
    }

    private void aimGunAt(EnemySnapshot target) {
        if (target == null) {
            return;
        }

        double bulletPower = 1.8;
        double bulletSpeed = 20.0 - (3.0 * bulletPower);
        double time = target.getDistance() / bulletSpeed;
        double futureX = target.getX() + Math.sin(target.getHeadingRadians()) * target.getVelocity() * time;
        double futureY = target.getY() + Math.cos(target.getHeadingRadians()) * target.getVelocity() * time;
        futureX = Math.max(18.0, Math.min(getBattleFieldWidth() - 18.0, futureX));
        futureY = Math.max(18.0, Math.min(getBattleFieldHeight() - 18.0, futureY));

        double aimBearing = Math.atan2(futureX - getX(), futureY - getY());
        setTurnGunRightRadians(Utils.normalRelativeAngle(aimBearing - getGunHeadingRadians()));
    }

    private void strafeTarget(EnemySnapshot target, double directionSign) {
        if (target == null) {
            moveTowardHeading(dangerMap.safestHeading(this, enemyManager.all()), 120.0);
            return;
        }

        double absBearing = target.absoluteBearingFrom(this);
        double desiredHeading = absBearing + (directionSign * (Math.PI / 2.0));
        desiredHeading += (target.getDistance() < 180.0 ? 0.35 : 0.0) * directionSign;
        moveTowardHeading(desiredHeading, target.getDistance() < 180.0 ? 120.0 : 80.0);
    }

    private void moveTowardHeading(double desiredHeading, double distance) {
        double angle = Utils.normalRelativeAngle(desiredHeading - getHeadingRadians());
        if (Math.abs(angle) > Math.PI / 2.0) {
            double reverseAngle = Utils.normalRelativeAngle(angle + Math.PI);
            setTurnRightRadians(reverseAngle);
            setBack(distance);
        } else {
            setTurnRightRadians(angle);
            setAhead(distance);
        }
    }

    private void fireIfAligned(EnemySnapshot target, double power) {
        if (target == null) {
            moveTowardHeading(dangerMap.safestHeading(this, enemyManager.all()), 100.0);
            return;
        }

        rewardTracker.onFireCommand();
        aimGunAt(target);
        if (getGunHeat() == 0.0
                && getEnergy() > power + 0.2
                && Math.abs(target.gunTurnFrom(this)) < Math.toRadians(8.0)) {
            setFire(power);
        } else {
            strafeTarget(target, lastAction == ActionType.STRAFE_LEFT ? -1.0 : 1.0);
        }
    }

    private void updateRadar(EnemySnapshot target) {
        EnemySnapshot staleEnemy = enemyManager.stalest(getTime());
        EnemySnapshot radarFocus = staleEnemy != null && staleEnemy.age(getTime()) > 6L ? staleEnemy : target;

        if (radarFocus == null) {
            if (getRadarTurnRemainingRadians() == 0.0) {
                setTurnRadarRightRadians(Double.POSITIVE_INFINITY);
            }
            return;
        }

        double absoluteBearing = radarFocus.absoluteBearingFrom(this);
        double radarTurn = Utils.normalRelativeAngle(absoluteBearing - getRadarHeadingRadians());
        double overshoot = Math.signum(radarTurn == 0.0 ? 1.0 : radarTurn) * Math.toRadians(22.0);
        setTurnRadarRightRadians(radarTurn + overshoot);
    }

    private double bulletDamage(double power) {
        return (4.0 * power) + Math.max(0.0, 2.0 * (power - 1.0));
    }
}
