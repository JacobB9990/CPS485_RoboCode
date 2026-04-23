package com.cps485.robocode.hybrid;

import robocode.util.Utils;

public final class Geometry {
    private Geometry() {
    }

    public static double distance(double x1, double y1, double x2, double y2) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return Math.hypot(dx, dy);
    }

    public static double absoluteBearing(double x1, double y1, double x2, double y2) {
        return Math.atan2(x2 - x1, y2 - y1);
    }

    public static double normalRelativeAngle(double angle) {
        return Utils.normalRelativeAngle(angle);
    }
}
