#!/usr/bin/env python3
"""channel_sounder: PN / Zadoff-Chu channel sounder (TX and RX) with power delay profile (PDP).

tx  : repeats a 1023-chip m-sequence (pn) or a Zadoff-Chu sequence (zc, constant amplitude) in one continuous stream.
rx  : records (iq_recorder) and analyses. ana: analyses a file.
Analysis: circular cross-correlation per period, power average over the periods (PDP), taps above --floor-db, mean
and rms delay spread, and the phase of the main tap per period (its slope gives the Doppler/CFO, see doppler_cfo.py).
Rule: TX and RX at the same center frequency and rate; CHEM CIR taps are in samples (L16), so the delays are
reported in samples and in microseconds at the record rate; wait 7-8 s after a CHEM cir change before a record.
"""
import argparse, json
import numpy as np
import vusrp_io as v
from siggen import mseq1023, zadoff_chu


def sequence(kind, zc_len=1021, zc_root=7):
    return mseq1023() if kind == "pn" else zadoff_chu(zc_len, zc_root)


def pdp(x, seq, rate, floor_db=-25.0, max_periods=400, ntaps=40):
    L = len(seq); K = min(len(x) // L, max_periods); S = np.conj(np.fft.fft(seq))
    acc = np.zeros(L); ph = []
    for k in range(K):
        c = np.fft.ifft(np.fft.fft(x[k * L:(k + 1) * L]) * S) / L
        acc += np.abs(c) ** 2; ph.append(c)
    acc /= K; i0 = int(np.argmax(acc)); prof = np.roll(acc, -i0) / acc[i0]
    # taps: delays 0..ntaps-1 after the main peak (pre-cursor taps wrap to the end of the period)
    pre = 8; p2 = np.r_[prof[-pre:], prof[:ntaps]]; d = np.arange(-pre, ntaps)
    keep = p2 > 10 ** (floor_db / 10)
    taps = [(int(dd), round(float(10 * np.log10(pp)), 2)) for dd, pp in zip(d[keep], p2[keep])]
    pw = p2[keep]; tau = d[keep] / rate
    mean_tau = float(np.sum(pw * tau) / np.sum(pw)); rms = float(np.sqrt(np.sum(pw * (tau - mean_tau) ** 2) / np.sum(pw)))
    phase = np.unwrap(np.angle([c[i0] for c in ph]))
    dop = float(np.polyfit(np.arange(K) * L / rate, phase, 1)[0] / (2 * np.pi)) if K > 2 else None
    noise = float(np.median(prof[ntaps:L - pre]));
    return {"periods": K, "period_samples": L, "peak_index": i0, "taps_samples_db": taps,
            "taps_us": [round(t / rate * 1e6, 3) for t, _ in taps], "mean_delay_us": round(mean_tau * 1e6, 3),
            "rms_delay_spread_us": round(rms * 1e6, 3), "pdp_noise_db": round(10 * np.log10(noise + 1e-20), 1),
            "main_tap_freq_hz": round(dop, 2) if dop is not None else None}, d, p2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["tx", "rx", "ana"]); p.add_argument("file", nargs="?", help="rx: output base; ana: input")
    p.add_argument("--seq", choices=["pn", "zc"], default="zc"); p.add_argument("--zc-len", type=int, default=1021); p.add_argument("--zc-root", type=int, default=7)
    p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=None, help="TX default 60, RX default 40")
    p.add_argument("--rms-dbfs", type=float, default=-18); p.add_argument("--secs", type=float, default=3)
    p.add_argument("--floor-db", type=float, default=-25); p.add_argument("--label", default="")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args(); seq = sequence(a.seq, a.zc_len, a.zc_root)
    if a.mode == "tx":
        tx = v.TX(a.rate, a.freq, a.gain or 60).start(v.scale_to_dbfs(seq, a.rms_dbfs), a.secs); tx.wait()
        print(json.dumps({"sent": tx.sent}), flush=True); v.hard_exit(0)
    if a.mode == "rx":
        from iq_recorder import record
        record(a.rate, a.freq, a.gain or 40, a.secs, a.file, a.label or f"sounder {a.seq}")
    x, rate, _, _ = v.read_iq(a.file, a.rate)
    r, d, p2 = pdp(x.astype(np.complex128), seq, rate, a.floor_db)
    r.update({"file": a.file, "seq": a.seq, "rate": rate, "label": a.label})
    json.dump(r, open(a.file + "_pdp.json", "w"), indent=1); print(json.dumps(r), flush=True)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(figsize=(5, 3))
        ax.stem(d / rate * 1e6, 10 * np.log10(p2 + 1e-12), bottom=-40); ax.set_ylim(-40, 3)
        ax.set_xlabel("delay (us)"); ax.set_ylabel("PDP (dB)"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(a.file + "_pdp.png", dpi=150)
    if a.mode == "rx":
        v.hard_exit(0)


if __name__ == "__main__":
    main()
