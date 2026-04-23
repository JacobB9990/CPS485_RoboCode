package com.cps485.robocode.hybrid;

import java.awt.Color;
import java.util.List;

import robocode.AdvancedRobot;
import robocode.BulletHitEvent;
import robocode.HitByBulletEvent;
import robocode.HitRobotEvent;
import robocode.HitWallEvent;
import robocode.RobotDeathEvent;
import robocode.ScannedRobotEvent;

public class HybridMeleeBot extends AdvancedRobot {
    private final EnemyTracker enemyTracker = new EnemyTracker();
    private final DangerMapBuilder dangerMapBuilder = new DangerMapBuilder(10, 10);
    private final TacticalManager tacticalManager = new RuleBasedTacticalManager();
    private final TargetSelector targetSelector = new WeightedTargetSelector();
    private final ModeAwareMovementController movementController = new ModeAwareMovementController();
    private final RadarController radarController = new SweepRadarController();
    private final GunController gunController = new GuessFactorGunController();

    private TacticalMode currentMode = TacticalMode.ENGAGE;

    @Override
    public void run() {
        setAdjustGunForRobotTurn(true);
        setAdjustRadarForGunTurn(true);
        setColors(new Color(34, 48, 74), new Color(230, 150, 32), new Color(245, 235, 210));

        while (true) {
            BotContext context = buildContext();
            EnemySnapshot target = targetSelector.selectTarget(context);
            currentMode = tacticalManager.chooseMode(context, target);

            movementController.apply(this, context, currentMode, target);
            gunController.apply(this, context, target, currentMode);
            radarController.apply(this, context, target);

            execute();
        }
    }

    @Override
    public void onScannedRobot(ScannedRobotEvent event) {
        enemyTracker.onScannedRobot(this, event);
    }

    @Override
    public void onRobotDeath(RobotDeathEvent event) {
        enemyTracker.onRobotDeath(event);
    }

    @Override
    public void onHitByBullet(HitByBulletEvent event) {
        movementController.reverseDirection();
    }

    @Override
    public void onHitWall(HitWallEvent event) {
        movementController.reverseDirection();
    }

    @Override
    public void onHitRobot(HitRobotEvent event) {
        movementController.reverseDirection();
    }

    @Override
    public void onBulletHit(BulletHitEvent event) {
        if (event.getEnergy() <= 0.0) {
            movementController.reverseDirection();
        }
    }

    private BotContext buildContext() {
        List<EnemySnapshot> enemies = enemyTracker.getAliveEnemies(getTime());
        BotContext partial = new BotContext(
                getTime(),
                getX(),
                getY(),
                getEnergy(),
                getVelocity(),
                getHeadingRadians(),
                getGunHeadingRadians(),
                getRadarHeadingRadians(),
                getBattleFieldWidth(),
                getBattleFieldHeight(),
                getOthers(),
                enemies,
                enemyTracker,
                null);
        DangerMap dangerMap = dangerMapBuilder.build(partial);
        return new BotContext(
                partial.getTime(),
                partial.getX(),
                partial.getY(),
                partial.getEnergy(),
                partial.getVelocity(),
                partial.getHeadingRadians(),
                partial.getGunHeadingRadians(),
                partial.getRadarHeadingRadians(),
                partial.getBattlefieldWidth(),
                partial.getBattlefieldHeight(),
                partial.getOthers(),
                partial.getEnemies(),
                partial.getEnemyTracker(),
                dangerMap);
    }
}
