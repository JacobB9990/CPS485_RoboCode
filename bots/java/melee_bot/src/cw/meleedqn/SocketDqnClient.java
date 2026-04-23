package cw.meleedqn;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Locale;

public final class SocketDqnClient {
    private final String host;
    private final int port;

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    public SocketDqnClient(String host, int port) {
        this.host = host;
        this.port = port;
    }

    public void connectIfNeeded() throws IOException {
        if (socket != null && socket.isConnected() && !socket.isClosed()) {
            return;
        }

        socket = new Socket(host, port);
        out = new PrintWriter(socket.getOutputStream(), true);
        in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
    }

    public int requestAction(double[] state, double reward, boolean done, BattleStats stats) throws IOException {
        StringBuilder payload = new StringBuilder();
        payload.append("STEP|")
                .append(format(reward)).append('|')
                .append(done ? 1 : 0).append('|')
                .append(stats.episode).append('|')
                .append(stats.tick).append('|')
                .append(stats.livingEnemies).append('|')
                .append(stats.placement).append('|')
                .append(stats.survivalTicks).append('|')
                .append(format(stats.damageDealt)).append('|')
                .append(format(stats.damageTaken)).append('|')
                .append(stats.kills).append('|')
                .append(stats.targetSwitches).append('|');

        for (int i = 0; i < state.length; i++) {
            if (i > 0) {
                payload.append(',');
            }
            payload.append(format(state[i]));
        }

        out.println(payload);
        String response = in.readLine();
        if (response == null || !response.startsWith("ACTION|")) {
            return ActionType.HEAD_TO_OPEN_SPACE.ordinal();
        }

        try {
            return Integer.parseInt(response.substring("ACTION|".length()).trim());
        } catch (NumberFormatException ignored) {
            return ActionType.HEAD_TO_OPEN_SPACE.ordinal();
        }
    }

    private String format(double value) {
        return String.format(Locale.US, "%.6f", value);
    }
}
