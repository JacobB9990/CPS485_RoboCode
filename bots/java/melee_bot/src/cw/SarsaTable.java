package cw;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

final class SarsaTable {
    private final int actionCount;
    private final Map<String, double[]> qValues = new HashMap<>();

    SarsaTable(int actionCount) {
        this.actionCount = actionCount;
    }

    double[] get(String state) {
        return qValues.computeIfAbsent(state, key -> new double[actionCount]);
    }

    void load(Path path) {
        if (!Files.exists(path)) {
            return;
        }

        try (BufferedReader reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty() || trimmed.startsWith("#")) {
                    continue;
                }

                String[] pieces = trimmed.split("\t", 2);
                if (pieces.length != 2) {
                    continue;
                }

                String[] valueParts = pieces[1].split(",");
                if (valueParts.length != actionCount) {
                    continue;
                }

                double[] values = new double[actionCount];
                for (int i = 0; i < actionCount; i++) {
                    values[i] = Double.parseDouble(valueParts[i]);
                }
                qValues.put(pieces[0], values);
            }
        } catch (IOException | NumberFormatException ignored) {
            // Start fresh if the table is missing or malformed.
        }
    }

    void save(Path path) {
        try {
            Path parent = path.getParent();
            if (parent != null) {
                Files.createDirectories(parent);
            }

            try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
                String[] states = qValues.keySet().toArray(new String[0]);
                Arrays.sort(states);

                for (String state : states) {
                    double[] values = qValues.get(state);
                    writer.write(state);
                    writer.write('\t');
                    for (int i = 0; i < values.length; i++) {
                        if (i > 0) {
                            writer.write(',');
                        }
                        writer.write(String.format(Locale.US, "%.8f", values[i]));
                    }
                    writer.newLine();
                }
            }
        } catch (IOException ignored) {
            // Ignore persistence failures during battle.
        }
    }
}
