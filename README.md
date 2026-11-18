# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Auspiciouspeony/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Auspiciouspeony/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A comparable-launch forecaster for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty: for each hour of release day it fits a kernel-weighted local regression of the log-ratio to hour 1 on two checkpoint features — log hour-1 viewers (launches whose hour 1 is already huge stay near flat, small ones explode later as regions unlock) and log viewers per live channel at hour 1 (a concentrated audience has less headroom for the evening peak) — weighting each past launch by how much it resembles GTA VI (rolling midnight-local unlock, console-only, no early access) and by how close its own hour-1 print is to this case's. The fitted centre is averaged 50/50 with the kernel-weighted median of the same ratios, and quantile spreads come from the fit's residual dispersion.

Leave-one-out backtest on the kit's comparable launches: loss **0.203** on big hour-1 launches (h1 ≥ 100k) vs 0.215 for the plain kernel median. On the kit's four practice cases: mean pinball loss **0.174** vs 0.190 for the plain kernel median (typical miss 32.5% vs 35.6%, 90% coverage 0.880 vs 0.870), house baseline ~0.387. Deterministic, standard library only, ~0.1 s a case.

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
