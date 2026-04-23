package com.cps485.robocode.hybrid;

public final class WeightedTargetSelector implements TargetSelector {
    @Override
    public EnemySnapshot selectTarget(BotContext context) {
        EnemySnapshot best = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        for (EnemySnapshot enemy : context.getEnemies()) {
            double score = scoreEnemy(enemy, context);
            if (score > bestScore) {
                bestScore = score;
                best = enemy;
            }
        }
        return best;
    }

    public double scoreEnemy(EnemySnapshot enemy, BotContext context) {
        double distanceScore = 350.0 / Math.max(100.0, enemy.getDistance());
        double lowEnergyBonus = Math.max(0.0, 40.0 - enemy.getEnergy()) * 0.08;
        double freshnessBonus = Math.max(0.0, 30.0 - enemy.age(context.getTime())) * 0.06;
        double crowdPenalty = context.getEnemyTracker().countNearby(enemy.getX(), enemy.getY(), 175.0, context.getTime()) * 0.45;
        double wallBonus = isNearWall(enemy, context) ? 0.55 : 0.0;

        return distanceScore + lowEnergyBonus + freshnessBonus + wallBonus - crowdPenalty;
    }

    private boolean isNearWall(EnemySnapshot enemy, BotContext context) {
        return enemy.getX() < 90.0
                || enemy.getY() < 90.0
                || enemy.getX() > context.getBattlefieldWidth() - 90.0
                || enemy.getY() > context.getBattlefieldHeight() - 90.0;
    }
}
