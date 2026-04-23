package com.cps485.robocode.hybrid;

public final class DangerMap {
    private final int rows;
    private final int cols;
    private final double cellWidth;
    private final double cellHeight;
    private final double[][] danger;

    public DangerMap(int rows, int cols, double battlefieldWidth, double battlefieldHeight) {
        this.rows = rows;
        this.cols = cols;
        this.cellWidth = battlefieldWidth / cols;
        this.cellHeight = battlefieldHeight / rows;
        this.danger = new double[rows][cols];
    }

    public void addDanger(int row, int col, double value) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return;
        }
        danger[row][col] += value;
    }

    public double getDangerAt(double x, double y) {
        int col = Math.min(cols - 1, Math.max(0, (int) (x / cellWidth)));
        int row = Math.min(rows - 1, Math.max(0, (int) (y / cellHeight)));
        return danger[row][col];
    }

    public Point2 pickSafestCellCenter(double robotX, double robotY) {
        double bestDanger = Double.POSITIVE_INFINITY;
        Point2 bestPoint = new Point2(robotX, robotY);

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                double x = (col + 0.5) * cellWidth;
                double y = (row + 0.5) * cellHeight;
                double distancePenalty = Geometry.distance(robotX, robotY, x, y) * 0.0025;
                double score = danger[row][col] + distancePenalty;
                if (score < bestDanger) {
                    bestDanger = score;
                    bestPoint = new Point2(x, y);
                }
            }
        }
        return bestPoint;
    }

    public static final class Point2 {
        public final double x;
        public final double y;

        public Point2(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }
}
