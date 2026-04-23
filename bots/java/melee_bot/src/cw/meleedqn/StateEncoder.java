package cw.meleedqn;

import java.util.Arrays;

import robocode.AdvancedRobot;
import robocode.util.Utils;

public final class StateEncoder {
    public static final int ENEMY_BLOCK_SIZE = 8;
    public static final int GLOBAL_FEATURE_COUNT = 16;
    public static final int STATE_SIZE = GLOBAL_FEATURE_COUNT + (ENEMY_BLOCK_SIZE * 4);

    public double[] encode(
            AdvancedRobot robot,
            EnemyManager enemyManager,
            TargetSelector selector,
            DangerMap dangerMap,
            RewardTracker rewardTracker,
            long tick
    ) {
        double[] state = new double[STATE_SIZE];
        double fieldWidth = robot.getBattleFieldWidth();
        double fieldHeight = robot.getBattleFieldHeight();
        double maxDistance = Math.hypot(fieldWidth, fieldHeight);

        int closeCount = dangerMap.countWithin(enemyManager.all(), 200.0);
        int mediumCount = dangerMap.countWithin(enemyManager.all(), 400.0) - closeCount;
        int farCount = Math.max(0, enemyManager.count() - closeCount - mediumCount);

        state[0] = clamp01(robot.getEnergy() / 100.0);
        state[1] = clampSigned(robot.getX() / fieldWidth);
        state[2] = clampSigned(robot.getY() / fieldHeight);
        state[3] = clamp01(robot.getX() / fieldWidth);
        state[4] = clamp01((fieldWidth - robot.getX()) / fieldWidth);
        state[5] = clamp01(robot.getY() / fieldHeight);
        state[6] = clamp01((fieldHeight - robot.getY()) / fieldHeight);
        state[7] = clamp01(robot.getOthers() / 12.0);
        state[8] = clamp01(enemyManager.closestDistance() / maxDistance);
        state[9] = clamp01(enemyManager.averageDistance() / maxDistance);
        state[10] = rewardTracker.wasRecentlyHit() ? 1.0 : 0.0;
        state[11] = clamp01(robot.getGunHeat() / 1.6);
        state[12] = dangerMap.crowdingScore(robot, enemyManager.all());
        state[13] = clamp01(closeCount / 10.0);
        state[14] = clamp01(mediumCount / 10.0);
        state[15] = clamp01(farCount / 10.0);

        EnemySnapshot nearest = enemyManager.nearest();
        EnemySnapshot weakest = enemyManager.weakest();
        EnemySnapshot threatening = selector.mostThreatening(robot, enemyManager.all(), tick);
        EnemySnapshot currentTarget = selector.getCurrentTargetName() == null
                ? null
                : enemyManager.get(selector.getCurrentTargetName());

        writeEnemyBlock(state, GLOBAL_FEATURE_COUNT, robot, nearest, tick, maxDistance);
        writeEnemyBlock(state, GLOBAL_FEATURE_COUNT + ENEMY_BLOCK_SIZE, robot, weakest, tick, maxDistance);
        writeEnemyBlock(state, GLOBAL_FEATURE_COUNT + (ENEMY_BLOCK_SIZE * 2), robot, threatening, tick, maxDistance);
        writeEnemyBlock(state, GLOBAL_FEATURE_COUNT + (ENEMY_BLOCK_SIZE * 3), robot, currentTarget, tick, maxDistance);
        return state;
    }

    private void writeEnemyBlock(
            double[] state,
            int offset,
            AdvancedRobot robot,
            EnemySnapshot enemy,
            long tick,
            double maxDistance
    ) {
        Arrays.fill(state, offset, offset + ENEMY_BLOCK_SIZE, 0.0);
        if (enemy == null) {
            return;
        }

        double absoluteBearing = enemy.absoluteBearingFrom(robot);
        double relativeBearing = Utils.normalRelativeAngle(absoluteBearing - robot.getHeadingRadians());

        state[offset] = Math.sin(relativeBearing);
        state[offset + 1] = Math.cos(relativeBearing);
        state[offset + 2] = clamp01(enemy.getDistance() / maxDistance);
        state[offset + 3] = Math.sin(enemy.getHeadingRadians());
        state[offset + 4] = Math.cos(enemy.getHeadingRadians());
        state[offset + 5] = clampSigned(enemy.getVelocity() / 8.0);
        state[offset + 6] = clamp01(enemy.getEnergy() / 100.0);
        state[offset + 7] = clamp01(Math.min(enemy.age(tick), 40L) / 40.0);
    }

    private double clamp01(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    private double clampSigned(double value) {
        return Math.max(-1.0, Math.min(1.0, (value * 2.0) - 1.0));
    }
}
