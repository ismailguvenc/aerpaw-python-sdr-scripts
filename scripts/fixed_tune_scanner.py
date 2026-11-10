#!/usr/bin/env python3
"""fixed_tune_scanner: multi-channel scanner inside one tuned band, by digital channelization (no retune).

Each channel (offset, bandwidth) is mixed to baseband, low-pass filtered and decimated (scipy resample_poly).
Output per channel: power time series in --dwell-ms steps, duty cycle against the noise floor, and optionally the
narrowband I/Q as SigMF. A "virtual stepped scan" visits the channels round-robin (one dwell each) on the same
record and shows what a retuning scanner (UHD1 style) would see, without a retune.
Rule: a V-USRP retune gives noise only (L17) and the RX must sit on the TX center frequency (L11), so all channels
must lie inside the one band at the fixed center frequency. Rate 1-23.04 MS/s.
"""
import argparse, json
import numpy as np
from scipy.signal import resample_poly
import vusrp_io as v


def channelize(x, rate, off, bw):
    D = max(1, int(rate // (1.25 * bw)))
    n = np.arange(len(x))
    y = resample_poly(x * np.exp(-2j * np.pi * off / rate * n), 1, D)
    return y.astype(np.complex64), rate / D


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?"); p.add_argument("--live", action="store_true")
    p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=40); p.add_argument("--secs", type=float, default=5)
    p.add_argument("--channels", default="-750e3:200e3,-250e3:200e3,250e3:200e3,750e3:200e3", help="'offset:bw,...' Hz")
    p.add_argument("--dwell-ms", type=float, default=2.0); p.add_argument("--thr-db", type=float, default=6.0)
    p.add_argument("--noise", help="noise-only capture for the per-channel floor")
    p.add_argument("--save-iq", action="store_true"); p.add_argument("--max-secs", type=float, default=10)
    p.add_argument("--out", default="scan"); p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.live:
        from iq_recorder import record
        record(a.rate, a.freq, a.gain, a.secs, a.out + "_iq", "fixed_tune_scanner"); a.file = a.out + "_iq"
    x, rate, freq, _ = v.read_iq(a.file, a.rate); x = x[:int(rate * a.max_secs)].astype(np.complex64)
    nz = v.read_iq(a.noise, rate)[0][:len(x)] if a.noise else None
    plan = [tuple(map(float, c.split(":"))) for c in a.channels.split(",")]
    for o, b in plan:
        if abs(o) + b / 2 > 0.5 * rate:
            raise v.RuleError(f"channel {o/1e6:+.3f} MHz outside the tuned band (no retune allowed)")
    series, res = [], []
    for i, (o, b) in enumerate(plan):
        y, r = channelize(x, rate, o, b); L = int(r * a.dwell_ms / 1e3); K = len(y) // L
        pw = np.mean(np.abs(y[:K * L].reshape(K, L)) ** 2, axis=1)
        if nz is not None:
            yn, _ = channelize(nz, rate, o, b); fl = float(np.median(np.abs(yn) ** 2))
        else:
            fl = float(np.percentile(pw, 5))
        occ = pw > fl * 10 ** (a.thr_db / 10)
        visits = occ[i::len(plan)]  # virtual stepped scan: channel i seen in dwell slot i, i+N, ...
        res.append({"offset_hz": o, "bw_hz": b, "out_rate": r, "floor_dbfs": round(10 * np.log10(fl + 1e-20), 2),
                    "mean_dbfs": round(float(10 * np.log10(pw.mean() + 1e-20)), 2), "duty_full": round(float(occ.mean()), 4),
                    "duty_stepped_scan": round(float(visits.mean()), 4), "scan_visits": int(len(visits))})
        series.append(10 * np.log10(pw + 1e-20))
        if a.save_iq:
            v.write_sigmf(f"{a.out}_ch{i}", y, r, (freq or 0) + o, f"channel {i} of {a.file}")
    out = {"file": a.file, "rate": rate, "dwell_ms": a.dwell_ms, "revisit_ms": a.dwell_ms * len(plan),
           "floor": "noise capture" if nz is not None else "p5 of dwell powers", "channels": res}
    json.dump(out, open(a.out + ".json", "w"), indent=1); print(json.dumps(out), flush=True)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(figsize=(6, 3))
        n = min(len(s) for s in series); t = np.arange(min(n, 100)) * a.dwell_ms
        for i, s in enumerate(series):
            ax.plot(t, s[:len(t)], label=f"{plan[i][0]/1e3:+.0f} kHz (duty {res[i]['duty_full']:.2f})")
        ax.set_xlabel("time (ms)"); ax.set_ylabel(f"power per {a.dwell_ms:g} ms (dBFS)"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper right")
        fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150)
    if a.live:
        v.hard_exit(0)


if __name__ == "__main__":
    main()
