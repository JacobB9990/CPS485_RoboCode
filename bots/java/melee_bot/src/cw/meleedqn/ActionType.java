package cw.meleedqn;

public enum ActionType {
    AHEAD_SHORT,
    AHEAD_MEDIUM,
    BACK_SHORT,
    BACK_MEDIUM,
    TURN_LEFT_SMALL,
    TURN_RIGHT_SMALL,
    TURN_LEFT_MEDIUM,
    TURN_RIGHT_MEDIUM,
    STRAFE_LEFT,
    STRAFE_RIGHT,
    HEAD_TO_OPEN_SPACE,
    FLEE_CLUSTER,
    FIRE_1,
    FIRE_2,
    FIRE_3;

    public static ActionType fromId(int id) {
        ActionType[] values = values();
        if (id < 0 || id >= values.length) {
            return HEAD_TO_OPEN_SPACE;
        }
        return values[id];
    }

    public static int count() {
        return values().length;
    }

    public boolean isFireAction() {
        return this == FIRE_1 || this == FIRE_2 || this == FIRE_3;
    }
}
