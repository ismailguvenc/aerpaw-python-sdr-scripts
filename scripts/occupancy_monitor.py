#!/usr/bin/env python3
"""occupancy_monitor: energy-detection spectrum occupancy per channel inside one captured band.

Input: a capture (SigMF or raw) or a live record (--live). Each block (default 1 ms) is windowed and transformed;
the power of the bins in each channel gives a time series. Noise floor: per channel from a noise-only capture
(--noise, best) or from the low percentile of the long-term median PSD (--floor-pct). A block is "occupied" when the
channel power is more than --thr-db above the channel noise power. Output: JSON (per channel: noise, mean, p90 power,
duty cycle), a CSV time series and an optional PNG (spectrogram + duty bars).
Rule: the monitor sees a transmitter only when its center frequency and rate equal those of the TX (L11), and CHEM
adds noise only while a TX is registered (L14): label the floor source. Rate 1-23.04 MS/s, fixed tune, skip 1 s.
"""
import argparse, csv, json
import numpy as np
import vusrp_io as v


def channel_plan(spec, rate, nchan, span):
    if spec:
        out = []
        for p in spec.split(","):
            o, b = p.split(":"); out.append((float(o), float(b)))
        return out
    bw = span / nchan
    return [(-span / 2 + bw * (i + 0.5), bw) for i in range(nchan)]


def block_psd(x, rate, block_ms):
    B = int(rate * block_ms / 1e3); K = len(x) // B
    w = np.hanning(B); w /= np.sqrt(np.mean(w ** 2))
    X = np.fft.fftshift(np.fft.fft(x[:K * B].reshape(K, B) * w, axis=1), axes=1)
    return np.abs(X) ** 2 / B ** 2, np.fft.fftshift(np.fft.fftfreq(B, 1 / rate))  # sum over bins = block power


def channel_power(P, f, plan):
    return np.stack([P[:, (f >= o - b / 2) & (f < o + b / 2)].sum(axis=1) for o, b in plan], axis=1)


def analyse(x, rate, plan, block_ms=1.0, thr_db=6.0, noise=None, floor_pct=10.0):
    P, f = block_psd(x, rate, block_ms); C = channel_power(P, f, plan)
    if noise is not None:
        Pn, _ = block_psd(noise, rate, block_ms); Cn = np.median(channel_power(Pn, f, plan), axis=0); src = "noise capture"
    else:
        med = np.median(P, axis=0); inner = np.abs(f) < 0.45 * rate
        dens = np.percentile(med[inner], floor_pct)
        Cn = np.array([dens * ((f >= o - b / 2) & (f < o + b / 2)).sum() for o, b in plan]); src = f"p{floor_pct:g} of median PSD"
    occ = C > Cn * 10 ** (thr_db / 10)
    ch = []
    for i, (o, b) in enumerate(plan):
        ch.append({"offset_hz": o, "bw_hz": b, "noise_dbfs": round(float(10 * np.log10(Cn[i] + 1e-20)), 2),
                   "mean_dbfs": round(float(10 * np.log10(C[:, i].mean() + 1e-20)), 2),
                   "p90_dbfs": round(float(10 * np.log10(np.percentile(C[:, i], 90) + 1e-20)), 2),
                   "duty": round(float(occ[:, i].mean()), 4)})
    return {"blocks": int(C.shape[0]), "block_ms": block_ms, "thr_db": thr_db, "floor": src, "channels": ch}, C, occ, P, f


def plot(path, P, f, rate, block_ms, res, title=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3), gridspec_kw={"width_ratios": [1.6, 1]})
    n = min(P.shape[0], 400); step = max(1, P.shape[1] // 512)
    S = 10 * np.log10(P[:n, ::step] + 1e-20)
    ax[0].imshow(S, aspect="auto", origin="lower", extent=[f[0] / 1e6, f[-1] / 1e6, 0, n * block_ms], vmin=np.percentile(S, 5), vmax=S.max(), cmap="viridis")
    ax[0].set_xlabel("offset (MHz)"); ax[0].set_ylabel("time (ms)"); ax[0].set_title(title or "spectrogram")
    d = [c["duty"] for c in res["channels"]]; lab = [f"{c['offset_hz']/1e6:+.2f}" for c in res["channels"]]
    ax[1].bar(range(len(d)), d, color="#3b6ea5"); ax[1].set_xticks(range(len(d))); ax[1].set_xticklabels(lab, rotation=45)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("duty cycle"); ax[1].set_xlabel("channel offset (MHz)")
    for i, y in enumerate(d):
        ax[1].text(i, y + 0.02, f"{y:.2f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="capture (SigMF base/meta or raw .c64); omit with --live")
    p.add_argument("--live", action="store_true", help="record --secs first (V-USRP)")
    p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=40); p.add_argument("--secs", type=float, default=5)
    p.add_argument("--channels", help="'offset:bw,...' in Hz"); p.add_argument("--nchan", type=int, default=4)
    p.add_argument("--span", type=float, help="band split into --nchan channels (default 0.8 x rate)")
    p.add_argument("--block-ms", type=float, default=1.0); p.add_argument("--thr-db", type=float, default=6.0)
    p.add_argument("--noise", help="noise-only capture for the floor"); p.add_argument("--floor-pct", type=float, default=10.0)
    p.add_argument("--max-secs", type=float, default=10); p.add_argument("--out", default="occupancy")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.live:
        from iq_recorder import record
        record(a.rate, a.freq, a.gain, a.secs, a.out + "_iq", "occupancy_monitor"); a.file = a.out + "_iq"
    x, rate, freq, _ = v.read_iq(a.file, a.rate); x = x[:int(rate * a.max_secs)]
    nz = v.read_iq(a.noise, rate)[0][:len(x)] if a.noise else None
    plan = channel_plan(a.channels, rate, a.nchan, a.span or 0.8 * rate)
    res, C, occ, P, f = analyse(x, rate, plan, a.block_ms, a.thr_db, nz, a.floor_pct)
    res.update({"file": a.file, "rate": rate, "freq": freq})
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["t_ms"] + [f"ch{i}_dbfs" for i in range(len(plan))] + [f"ch{i}_occ" for i in range(len(plan))])
        for k in range(C.shape[0]):
            w.writerow([round(k * a.block_ms, 3)] + [round(float(10 * np.log10(c + 1e-20)), 2) for c in C[k]] + [int(o) for o in occ[k]])
    if a.plot:
        plot(a.out + ".png", P, f, rate, a.block_ms, res)
    print(json.dumps(res), flush=True)
    if a.live:
        v.hard_exit(0)


if __name__ == "__main__":
    main()
