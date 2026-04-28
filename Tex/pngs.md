I’m locating the figure sources and captions first so I can explain them in the right context rather than guessing from the file names.

Read memory [](file:///Users/jacobbecker/Library/Application%20Support/Code/User/workspaceStorage/db07c7b0e67ba9e71fe94df8ad736989/GitHub.copilot-chat/memory-tool/memories/repo/dqn-notes.md)

Search complete: Find figure sources

Searching codebase for "figure references captions LaTeX presentation report docs"

Searched codebase for "figures I got experimental results training curves evaluation charts", 25 results

Searched for files matching `**/Tex/**`, 7 matches

Searched for files matching `**/docs/**`, 9 matches

Searched for regex `figure.*caption|caption.*figure|\[figure\]|\[Fig\]|Figure\s*\(`, 3 results

Searching for regex `\includegraphics|\addfig|\include{.*fig}|plot.*figure|figure.*output`

Searched for regex `training.*curve|loss.*curve|accuracy.*chart|evaluation.*plot|result.*figure`, 70 results

Searched for files matching `**/*.tex`, 2 matches

Searched for regex `figure|Figure|figcaption|caption`, 22 results

Searching for regex `\includegraphics|\addfig|\include{.*fig}|plot.*figure|figure.*output`

Searched for files matching `**/*.pdf`, 4 matches

Searched for files matching `**/*.md`, 17 matches

Searched for regex `training.*curve|loss.*curve|accuracy.*chart|evaluation.*plot|result.*figure`, 70 results

Read [](file:///Users/jacobbecker/CPS485_RoboCode/Tex/robocode_ai_final.tex#1-1), lines 1 to 260

Read [](file:///Users/jacobbecker/CPS485_RoboCode/Tex/presentation_concrete_notes.md#1-1), lines 1 to 260

Read [](file:///Users/jacobbecker/CPS485_RoboCode/Tex/robocode_ai_presentation.tex#1-1), lines 1 to 260

Read [](file:///Users/jacobbecker/CPS485_RoboCode/scripts/analyze/generate_latex_report_prompt.py)

Read [](file:///Users/jacobbecker/CPS485_RoboCode/scripts/analyze/summarize_results.py)

I’ve found the slide deck and analysis pipeline; next I’m checking the actual figure outputs so I can explain each one by what it measures and how to read it.

Searched for files matching `CPS485_RoboCode` · `logs/analysis/figures/*.png`, 4 matches

Searched for files matching `CPS485_RoboCode` · `logs/analysis/*.json`, 2 matches

Read [](file:///Users/jacobbecker/CPS485_RoboCode/Tex/robocode_ai_final.tex#220-220), lines 220 to 420

The figures are the four analysis plots from the summary pipeline. I’m pulling the underlying summary data now so I can describe what each plot is actually showing, not just the filename.

Read [](file:///Users/jacobbecker/CPS485_RoboCode/Tex/robocode_ai_final.tex#420-420), lines 420 to 470

Read [](file:///Users/jacobbecker/CPS485_RoboCode/logs/analysis/summary_20260428_091712.json#1-1), lines 1 to 260

I have the summary values; I’m checking one or two plots visually so I don’t miss the orientation or legend order when I explain them.

Viewed image [](file:///Users/jacobbecker/CPS485_RoboCode/logs/analysis/figures/win_rate_bar_chart.png)

Viewed image [](file:///Users/jacobbecker/CPS485_RoboCode/logs/analysis/figures/placement_heatmap.png)

The win-rate and placement plots confirm the headline story. I’m checking the training curves now so I can describe the learning dynamics accurately too.

Viewed image [](file:///Users/jacobbecker/CPS485_RoboCode/logs/analysis/figures/reward_curves.png)

Viewed image [](file:///Users/jacobbecker/CPS485_RoboCode/logs/analysis/figures/fitness_curves.png)

These are the four analysis plots generated from the log summary in logs/analysis/summary_20260428_091712.json and embedded in the slide deck in Tex/robocode_ai_final.tex.

- win_rate_bar_chart.png shows win rate by bot and opponent set. The main takeaway is that Jacob3_0 is the strongest overall, MeleeDQN is far behind, PPO is mixed, and SARSA sits in the middle.
- reward_curves.png shows episode reward over training. What matters here is the trend: Jacob3_0 stays closest to positive territory, while MeleeDQN, PPO, and SARSA are much noisier and often negative, which means they were learning but not consistently turning that into good battle outcomes.
- fitness_curves.png is for NeuroEvoMelee. Both best fitness and mean fitness rise over generations, so evolution is improving the population, but that does not automatically mean the bot is winning matches yet.
- placement_heatmap.png shows finish-position counts. Brighter cells mean that bot finished in that placement more often. Jacob3_0 dominates the top placements, while the other bots spend more time in lower or middle positions.

If you want, I can also turn these into a presentation-ready script, one figure at a time, so you can explain them out loud.