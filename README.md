# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Auspiciouspeony/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Auspiciouspeony/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A comparable-launch forecaster for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty: for each hour of release day it takes the median log-ratio to hour 1 across the case's past launches, weighting each launch by how much it resembles GTA VI (rolling midnight-local unlock, console-only, no early access) and by how close its own hour-1 print is to this case's — a launch whose hour 1 is already huge stays near flat, one with a small hour 1 explodes later as regions unlock, and the kernel picks the right family instead of averaging both. Quantile spreads come from the same cross-launch dispersion.

Leave-one-out backtest on the kit's comparable launches (console-only, midnight-local family): mean pinball loss **0.17** vs ~0.30 for the house baseline on the same launches. Deterministic, standard library only, ~0.1 s a case.

## Run it

```sh
python3 main.py
```

With the kit's practice cases:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=wa0stw5e&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
