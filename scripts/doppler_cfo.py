#!/usr/bin/env python3
"""doppler_cfo: frequency offset (CFO + Doppler) of a pilot tone over time, compared with the CHEM link log.

Input: a tone record from iq_recorder (siggen tone at --f0) and its <base>_chem.csv. Per window (default 100 ms):
Hann window, zero-padded FFT, peak search within +-search Hz of f0, quadratic interpolation. Output: JSON with the
offset time series, CHEM freqOffsetHz + dopplerHz at the same times, the error after each CHEM change has acted,
and the time from a CHEM change to the measured step. Optional PNG.
Rule: TX and RX at one center frequency (the twin adds no CFO of its own, L21); one Doppler shift per link (L5);
a CHEM fo change acts after about 7 s (L6), so compare only settled segments.
"""
import argparse, datetime, json, os
import numpy as np
import vusrp_io as v


def tone_track(x, rate, f0, win_s=0.1, search=5e3, pad=8):
    W = int(rate * win_s); n = W * pad; w = np.hanning(W); f = np.fft.fftfreq(n, 1 / rate)
    band = np.abs(f - f0) <= search; idx = np.where(band)[0]
    t, off = [], []
    for k in range(len(x) // W):
        X = np.abs(np.fft.fft(x[k * W:(k + 1) * W] * w, n)) ** 2
        i = idx[np.argmax(X[idx])]; a, b, c = np.log(X[(i - 1) % n] + 1e-30), np.log(X[i] + 1e-30), np.log(X[(i + 1) % n] + 1e-30)
        d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
        t.append((k + 0.5) * win_s); off.append((f[i] + d * rate / n) - f0)
    return np.array(t), np.array(off)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="iq_recorder base (SigMF)"); p.add_argument("--f0", type=float, default=100e3)
    p.add_argument("--chem", help="CHEM CSV (default <file>_chem.csv)"); p.add_argument("--win", type=float, default=0.1)
    p.add_argument("--search", type=float, default=5e3); p.add_argument("--settle", type=float, default=1.0,
                   help="seconds after a measured step that are left out of the error")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    x, rate, freq, meta = v.read_iq(a.file)
    t, off = tone_track(x.astype(np.complex128), rate, a.f0, a.win, a.search)
    t0 = datetime.datetime.strptime(meta["captures"][0]["core:datetime"], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    cp = a.chem or a.file + "_chem.csv"
    rows = v.load_chem_csv(cp) if os.path.exists(cp) else []
    ct = np.array([r["t"] - t0 for r in rows]); cfo = np.array([float(r.get("freqOffsetHz") or 0) for r in rows])
    cdop = np.array([float(r.get("dopplerHz") or 0) if str(r.get("dopplerEnabled")) == "True" else 0.0 for r in rows])
    chem_tot = np.interp(t, ct, cfo + cdop) if len(rows) else np.zeros_like(t)
    # CHEM-reported changes and the measured response
    changes = []
    for i in range(1, len(ct)):
        if abs((cfo + cdop)[i] - (cfo + cdop)[i - 1]) > 1:
            changes.append({"chem_t_s": round(float(ct[i]), 2), "from_hz": float((cfo + cdop)[i - 1]), "to_hz": float((cfo + cdop)[i])})
    for c in changes:
        mid = (c["from_hz"] + c["to_hz"]) / 2; after = np.where((t > c["chem_t_s"] - 10) & ((off - mid) * np.sign(c["to_hz"] - c["from_hz"]) > 0))[0]
        c["measured_step_t_s"] = round(float(t[after[0]]), 2) if len(after) else None
        c["lag_s"] = round(c["measured_step_t_s"] - c["chem_t_s"], 2) if len(after) else None
    # error where the measured offset is settled near the CHEM value (CHEM reports the setting before it acts)
    ok = np.ones_like(t, bool)
    jumps = np.where(np.abs(np.diff(off)) > 20)[0]
    for j in jumps:
        ok &= ~((t >= t[j]) & (t <= t[j] + a.settle))
    err = off[ok] - chem_tot[ok]; settled = np.abs(err) < 50
    seg = []
    for val in sorted(set(np.round(chem_tot, 1))):
        m = ok & (np.abs(off - val) < 50)
        if m.sum() > 3:
            seg.append({"chem_hz": float(val), "measured_median_hz": round(float(np.median(off[m])), 2),
                        "measured_std_hz": round(float(np.std(off[m])), 2), "windows": int(m.sum())})
    if not rows:
        seg = [{"chem_hz": None, "measured_median_hz": round(float(np.median(off)), 2), "measured_std_hz": round(float(np.std(off)), 2), "windows": len(t)}]
    r = {"file": a.file, "windows": len(t), "win_s": a.win, "chem_rows": len(rows), "segments": seg, "chem_changes": changes,
         "settled_abs_err_median_hz": round(float(np.median(np.abs(err[settled]))), 3) if settled.any() else None}
    json.dump({**r, "t": t.round(3).tolist(), "offset_hz": off.round(2).tolist()}, open(a.file + "_cfo.json", "w"))
    print(json.dumps(r), flush=True)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(t, off, ".", ms=3, label="measured tone offset")
        if len(rows):
            ax.step(ct, cfo + cdop, where="post", label="CHEM fo + Doppler (setting)")
        ax.set_xlim(0, t[-1] + a.win); ax.set_xlabel("time in record (s)"); ax.set_ylabel("frequency offset (Hz)"); ax.grid(alpha=.3); ax.legend(fontsize=9)
        fig.tight_layout(); fig.savefig(a.file + "_cfo.png", dpi=150)


if __name__ == "__main__":
    main()
