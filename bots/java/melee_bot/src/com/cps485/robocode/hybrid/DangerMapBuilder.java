package com.cps485.robocode.hybrid;

import java.util.List;

public final class DangerMapBuilder {
    private final int rows;
    private final int cols;

    public DangerMapBuilder(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
    }

    public DangerMap build(BotContext context) {
        DangerMap map = new DangerMap(rows, cols, context.getBattlefieldWidth(), context.getBattlefieldHeight());
        List<EnemySnapshot> enemies = context.getEnemies();

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double x = (col + 0.5) * (context.getBattlefieldWidth() / cols);
                double y = (row + 0.5) * (context.getBattlefieldHeight() / rows);
                double risk = wallRisk(x, y, context);

                for (EnemySnapshot enemy : enemies) {
                    double distance = Geometry.distance(x, y, enemy.getX(), enemy.getY());
                    risk += (enemy.getEnergy() + 20.0) / Math.max(75.0, distance);
                }

                int crowdCount = context.getEnemyTracker().countNearby(x, y, 200.0, context.getTime());
                risk += crowdCount * 1.25;
                map.addDanger(row, col, risk);
            }
        }

        return map;
    }

    private double wallRisk(double x, double y, BotContext context) {
        double margin = 80.0;
        double left = Math.max(1.0, x);
        double right = Math.max(1.0, context.getBattlefieldWidth() - x);
        double bottom = Math.max(1.0, y);
        double top = Math.max(1.0, context.getBattlefieldHeight() - y);

        double risk = 0.0;
        risk += margin / left;
        risk += margin / right;
        risk += margin / bottom;
        risk += margin / top;
        return risk;
    }
}
