import dev.robocode.tankroyale.runner.BattleResults;
import dev.robocode.tankroyale.runner.BattleRunner;
import dev.robocode.tankroyale.runner.BattleSetup;
import dev.robocode.tankroyale.runner.BotEntry;
import dev.robocode.tankroyale.runner.BotResult;

import java.nio.file.Path;
import java.time.Duration;
import java.util.List;

public class RunBattle {
    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: RunBattle <botA_dir> <botB_dir> <rounds> <port>");
            System.exit(2);
        }

        String botA = args[0];
        String botB = args[1];
        int rounds = Integer.parseInt(args[2]);
        int port = Integer.parseInt(args[3]);

        BattleSetup setup = BattleSetup.classic(builder -> {
            builder.setNumberOfRounds(rounds);
            builder.setMaxNumberOfParticipants(2);
            builder.setReadyTimeoutMicros(3_000_000);
            builder.setTurnTimeoutMicros(30_000);
            builder.setMaxInactivityTurns(450);
        });

        List<BotEntry> bots = List.of(
                BotEntry.of(Path.of(botA)),
                BotEntry.of(Path.of(botB))
        );

        try (BattleRunner runner = BattleRunner.create(builder -> {
            if (port > 0) {
                builder.embeddedServer(port);
            } else {
                builder.embeddedServer();
            }
            builder.enableServerOutput();
            builder.botConnectTimeout(Duration.ofSeconds(60));
        })) {
            BattleResults results = runner.runBattle(setup, bots);
            System.out.println("Rounds played: " + results.getNumberOfRounds());
            for (BotResult r : results.getResults()) {
                System.out.printf("#%d %s v%s score=%d first=%d\n",
                        r.getRank(), r.getName(), r.getVersion(), r.getTotalScore(), r.getFirstPlaces());
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }
}
