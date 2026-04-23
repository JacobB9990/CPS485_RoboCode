import dev.robocode.tankroyale.runner.BattleResults;
import dev.robocode.tankroyale.runner.BattleRunner;
import dev.robocode.tankroyale.runner.BattleSetup;
import dev.robocode.tankroyale.runner.BotEntry;
import dev.robocode.tankroyale.runner.BotResult;

import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

public class RunMeleeBattle {
    public static void main(String[] args) {
        if (args.length < 4) {
            System.err.println("Usage: RunMeleeBattle <rounds> <port> <bot_dir> <bot_dir> [<bot_dir>...]");
            System.exit(2);
        }

        int rounds = Integer.parseInt(args[0]);
        int port = Integer.parseInt(args[1]);

        List<BotEntry> bots = new ArrayList<>();
        for (int i = 2; i < args.length; i++) {
            bots.add(BotEntry.of(Path.of(args[i])));
        }

        BattleSetup setup = BattleSetup.classic(builder -> {
            builder.setNumberOfRounds(rounds);
            builder.setMaxNumberOfParticipants(bots.size());
            builder.setReadyTimeoutMicros(3_000_000);
            builder.setTurnTimeoutMicros(30_000);
            builder.setMaxInactivityTurns(600);
        });

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
            System.out.println(toJson(results));
        } catch (Exception e) {
            e.printStackTrace(System.err);
            System.exit(1);
        }
    }

    private static String toJson(BattleResults results) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"rounds\":").append(results.getNumberOfRounds()).append(",\"results\":[");

        List<BotResult> botResults = results.getResults();
        for (int i = 0; i < botResults.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            BotResult r = botResults.get(i);
            sb.append('{')
                    .append("\"name\":\"").append(jsonEscape(r.getName())).append("\",")
                    .append("\"version\":\"").append(jsonEscape(r.getVersion())).append("\",")
                    .append("\"rank\":").append(r.getRank()).append(',')
                    .append("\"totalScore\":").append(r.getTotalScore()).append(',')
                    .append("\"survivalScore\":").append(r.getSurvival()).append(',')
                    .append("\"lastSurvivorBonus\":").append(r.getLastSurvivorBonus()).append(',')
                    .append("\"bulletDamageScore\":").append(r.getBulletDamage()).append(',')
                    .append("\"bulletKillBonus\":").append(r.getBulletKillBonus()).append(',')
                    .append("\"ramDamageScore\":").append(r.getRamDamage()).append(',')
                    .append("\"ramKillBonus\":").append(r.getRamKillBonus()).append(',')
                    .append("\"firstPlaces\":").append(r.getFirstPlaces()).append(',')
                    .append("\"secondPlaces\":").append(r.getSecondPlaces()).append(',')
                    .append("\"thirdPlaces\":").append(r.getThirdPlaces())
                    .append('}');
        }

        sb.append("]}");
        return sb.toString();
    }

    private static String jsonEscape(String value) {
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"");
    }
}
