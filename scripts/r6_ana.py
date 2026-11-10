#!/usr/bin/env python3
"""r6_ana: small offline analyses for Report-5 (CPU only, no radio).

  r6_ana.py step <sigmf_base> [--block-ms 10]
      Block power (dBFS) against the absolute wall time of the record (SigMF core:datetime). JSON to <base>_step.json.
  r6_ana.py fold <sigmf_base> [--noise <sigmf_base>] [--nchan 8] [--span 8.64e6] [--thr-db 10] [--period-blocks 10]
      occupancy_monitor analysis of an NR record in 0.5 ms blocks (one 30 kHz slot), then a fold of the occupied
      blocks over the TDD period (10 slots = 5 ms). The phase of the fold is arbitrary (no slot timing at the monitor):
      the profile is rotated so that its largest rising edge of power is slot 0. JSON to <base>_fold.json.
  r6_ana.py lag <sigmf_base>
      Arrival lag of the samples from <base>_lag.npy (iq_recorder.py --lag-log): lag = wall - (t0 + n/rate), median removed.
      Counts steps above 0.4 ms and gives the phase of each step start inside the wall-clock second. JSON to <base>_lagana.json.
  r6_ana.py bsnr <sigmf_base>
      SNR of a bursty signal in 0.5 ms blocks: on-blocks (top 25 %), off-blocks (bottom 50 %), full-band SNR during the bursts
      and the average-power SNR. JSON to <base>_bsnr.json.
  r6_ana.py psd <sigmf_base> [--nfft 4096]
      Welch PSD (Hann, 50 % overlap) in dB per bin, relative to full scale. JSON to <base>_psd.json.
"""
import argparse, datetime, json
import numpy as np
import vusrp_io as v
import occupancy_monitor as om


def t0_of(meta):
    s = meta["captures"][0]["core:datetime"].replace("Z", "+00:00")
    return datetime.datetime.fromisoformat(s).timestamp()


def step(a):
    x, rate, freq, meta = v.read_iq(a.base)
    B = int(rate * a.block_ms / 1e3); K = len(x) // B
    p = 10 * np.log10(np.mean(np.abs(x[:K * B].reshape(K, B)) ** 2, axis=1) + 1e-20)
    out = {"file": a.base, "t0": t0_of(meta), "block_s": a.block_ms / 1e3, "dbfs": [round(float(y), 3) for y in p]}
    json.dump(out, open(a.base + "_step.json", "w"))
    print(json.dumps({"blocks": K, "median": round(float(np.median(p)), 2), "min": round(float(p.min()), 2), "max": round(float(p.max()), 2)}))


def fold(a):
    x, rate, freq, meta = v.read_iq(a.base)
    nz = v.read_iq(a.noise)[0] if a.noise else None
    plan = om.channel_plan(None, rate, a.nchan, a.span)
    res, C, occ, P, f = om.analyse(x, rate, plan, 0.5, a.thr_db, nz)
    tot = C.sum(axis=1)                                     # in-band power per block (all channels)
    K = (len(tot) // a.period_blocks) * a.period_blocks
    prof_p = 10 * np.log10(tot[:K].reshape(-1, a.period_blocks).mean(axis=0))
    prof_o = occ[:K].mean(axis=1).reshape(-1, a.period_blocks).mean(axis=0)   # mean share of channels occupied
    r = int(np.argmax(prof_p - np.roll(prof_p, 1)))         # rotate: the largest rising edge is slot 0
    res.update({"file": a.base, "rate": rate, "freq": freq, "t0": t0_of(meta),
                "band_duty": round(float(occ.mean()), 4),
                "fold_power_dbfs": [round(float(y), 2) for y in np.roll(prof_p, -r)],
                "fold_occ": [round(float(y), 3) for y in np.roll(prof_o, -r)],
                "excerpt_power_dbfs": [round(float(y), 2) for y in 10 * np.log10(tot[:200] + 1e-20)],
                "stats": meta["global"].get("aerpaw:stats")})
    json.dump(res, open(a.base + "_fold.json", "w"))
    print(json.dumps({"band_duty": res["band_duty"], "duty": [c["duty"] for c in res["channels"]], "fold_occ": res["fold_occ"]}))


def psd(a):
    x, rate, freq, meta = v.read_iq(a.base)
    N = a.nfft; w = np.hanning(N); hop = N // 2; K = (len(x) - N) // hop
    acc = np.zeros(N)
    for k in range(K):
        acc += np.abs(np.fft.fft(x[k * hop:k * hop + N] * w)) ** 2
    P = np.fft.fftshift(acc / K) / (N * np.sum(w ** 2))   # sum over bins = mean power
    f = np.fft.fftshift(np.fft.fftfreq(N, 1 / rate))
    pk = int(np.argmax(P))
    out = {"file": a.base, "rate": rate, "freq": freq, "nfft": N, "segments": K, "f_hz": [round(float(y), 1) for y in f],
           "psd_db": [round(float(y), 2) for y in 10 * np.log10(P + 1e-20)], "peak_hz": round(float(f[pk]), 1),
           "rms_dbfs": round(v.dbfs(x), 2), "stats": meta["global"].get("aerpaw:stats")}
    json.dump(out, open(a.base + "_psd.json", "w"))
    print(json.dumps({"segments": K, "peak_hz": out["peak_hz"], "rms_dbfs": out["rms_dbfs"], "median_db": round(float(np.median(out["psd_db"])), 2)}))


def lag(a):
    meta = json.load(open(a.base + ".sigmf-meta"))
    rate = float(meta["global"]["core:sample_rate"])
    L = np.load(a.base + "_lag.npy")[5:]; wall, n = L[:, 0], L[:, 1]        # the first reads hold the start of the stream
    t0 = float(np.median(wall - n / rate)); lg = wall - (t0 + n / rate)
    lg = lg - np.median(lg)
    d = np.diff(lg); idx = np.where(d > 4e-4)[0] + 1
    steps = []
    for i in idx:
        j = i
        while j < len(lg) - 1 and lg[j] > 1e-4 and wall[j] - wall[i] < 2.0:
            j += 1
        steps.append({"t_s": round(float(wall[i] - wall[0]), 3), "step_ms": round(float(d[i - 1] * 1e3), 3), "peak_ms": round(float(lg[i:j + 1].max() * 1e3), 3),
                      "decay_ms": round(float((wall[j] - wall[i]) * 1e3), 1), "phase_s": round(float(wall[i] % 1.0), 3)})
    ch = max(1, len(wall) // 4000)
    small = [q for q in steps if q["peak_ms"] < 1.5]; large = [q for q in steps if q["peak_ms"] >= 1.5]
    ph = np.array([q["phase_s"] for q in steps]) if steps else np.array([])
    hist = np.histogram(ph, bins=20, range=(0, 1))[0].tolist() if len(ph) else [0] * 20
    med = lambda k, q: round(float(np.median([x[k] for x in q])), 2) if q else None
    out = {"file": a.base, "rate": rate, "reads": int(len(wall)), "span_s": round(float(wall[-1] - wall[0]), 2),
           "lag_ms_min": round(float(lg.min() * 1e3), 3), "lag_ms_max": round(float(lg.max() * 1e3), 3), "lag_ms_std": round(float(lg.std() * 1e3), 3),
           "n_steps_gt_0p4ms": len(steps), "n_small_lt_1p5ms": len(small), "n_large_ge_1p5ms": len(large),
           "small_peak_ms_range": [min(x["peak_ms"] for x in small), max(x["peak_ms"] for x in small)] if small else None,
           "large_peak_ms_range": [min(x["peak_ms"] for x in large), max(x["peak_ms"] for x in large)] if large else None,
           "small_decay_ms_median": med("decay_ms", small), "large_decay_ms_median": med("decay_ms", large),
           "steps": steps[:200], "phase_hist_20": hist,
           "t_s": [round(float(y), 4) for y in (wall - wall[0])[:(len(wall) // ch) * ch:ch]],
           "lag_ms": [round(float(y * 1e3), 4) for y in lg[:(len(lg) // ch) * ch].reshape(-1, ch).max(axis=1)],
           "t_s_steps": [q["t_s"] for q in steps], "peak_ms_steps": [q["peak_ms"] for q in steps]}
    json.dump(out, open(a.base + "_lagana.json", "w"))
    print(json.dumps({k: out[k] for k in ("reads", "span_s", "lag_ms_max", "lag_ms_std", "n_steps_gt_0p4ms", "n_small_lt_1p5ms", "n_large_ge_1p5ms",
                                          "small_peak_ms_range", "large_peak_ms_range", "small_decay_ms_median", "large_decay_ms_median", "phase_hist_20")}))


def bsnr(a):
    x, rate, freq, meta = v.read_iq(a.base)
    B = int(rate * 0.5e-3); K = len(x) // B
    p = np.mean(np.abs(x[:K * B].reshape(K, B)) ** 2, axis=1)
    o = np.sort(p); P_on = float(np.mean(o[-K // 4:])); P_off = float(np.mean(o[:K // 2])); P_mean = float(np.mean(p))
    out = {"file": a.base, "blocks": K, "p_on_dbfs": round(10 * np.log10(P_on), 2), "p_off_dbfs": round(10 * np.log10(P_off), 2),
           "snr_on_db": round(float(10 * np.log10((P_on - P_off) / P_off)), 2),
           "snr_avg_db": round(float(10 * np.log10((P_mean - P_off) / P_off)), 2),
           "block_dbfs_first_80": [round(float(10 * np.log10(y + 1e-20)), 2) for y in p[:80]]}
    json.dump(out, open(a.base + "_bsnr.json", "w"))
    print(json.dumps({k: out[k] for k in ("blocks", "p_on_dbfs", "p_off_dbfs", "snr_on_db", "snr_avg_db")}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["step", "fold", "psd", "lag", "bsnr"]); p.add_argument("base")
    p.add_argument("--block-ms", type=float, default=10.0); p.add_argument("--noise")
    p.add_argument("--nchan", type=int, default=8); p.add_argument("--span", type=float, default=8.64e6)
    p.add_argument("--thr-db", type=float, default=10.0); p.add_argument("--period-blocks", type=int, default=10); p.add_argument("--nfft", type=int, default=4096)
    a = p.parse_args()
    {"step": step, "fold": fold, "psd": psd, "lag": lag, "bsnr": bsnr}[a.mode](a)


if __name__ == "__main__":
    main()
