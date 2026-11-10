#!/usr/bin/env python3
"""dl_capture: capture an eNB/gNB downlink at its own rate for offline decoding, with a PSS correlation check.

capture : records the DL (iq_recorder: SigMF + CHEM log) at the cell center frequency and sample rate.
check   : PSS search on a record. NR: the three m-sequence PSS (NID2 0-2) at the SSB position (--ssb-offset), FFT size
          rate/SCS. LTE: the three Zadoff-Chu PSS (roots 25, 29, 34), FFT size rate/15 kHz. Gives the best NID2, the
          normalized correlation peak, the peak count and the median peak spacing (NR SSB 20 ms, LTE PSS 5 ms).
nrsib1  : cuts the record into windows and runs the Rust nr-sib1 decoder (github.com/zackiles/nr-sib1, examples/decode)
          on each window (SSB sync, MIB, SIB1). Needs the decoder binary (--decoder).
Rule: RX rate = cell rate (TX and RX need the same rate, L11), rate at most 23.04 MS/s (use 10-MHz cells), RX center =
cell center (no LO offset), skip the first second, start the capture after the cell runs.
"""
import argparse, json, os, subprocess
import numpy as np
from scipy.signal import fftconvolve
import vusrp_io as v


def nr_pss(nid2, nfft):
    x = [0, 1, 1, 0, 1, 1, 1]
    for i in range(127 - 7):
        x.append((x[i + 4] + x[i]) % 2)
    d = np.array([1 - 2 * x[(n + 43 * nid2) % 127] for n in range(127)], complex)
    X = np.zeros(nfft, complex); k = np.arange(127) - 64  # PSS on SSB subcarriers 56..182, SSB center = subcarrier 120
    X[k % nfft] = d
    return np.fft.ifft(X)


def lte_pss(nid2, nfft):
    u = (25, 29, 34)[nid2]; n = np.arange(62)
    d = np.where(n < 31, np.exp(-1j * np.pi * u * n * (n + 1) / 63), np.exp(-1j * np.pi * u * (n + 1) * (n + 2) / 63))
    X = np.zeros(nfft, complex); k = np.r_[-31:0, 1:32]
    X[k % nfft] = d
    return np.fft.ifft(X)


def pss_check(x, rate, rat, scs=30e3, ssb_offset=0.0, thr=0.3):
    nfft = int(round(rate / (scs if rat == "nr" else 15e3)))
    if ssb_offset:
        x = x * np.exp(-2j * np.pi * ssb_offset / rate * np.arange(len(x)))
    e = np.convolve(np.abs(x) ** 2, np.ones(nfft), "valid")
    best = None
    for nid2 in range(3):
        t = nr_pss(nid2, nfft) if rat == "nr" else lte_pss(nid2, nfft)
        c = np.abs(fftconvolve(x, np.conj(t[::-1]), "valid")) ** 2 / (e[:len(x) - nfft + 1] * np.sum(np.abs(t) ** 2) + 1e-30)
        pk = float(c.max())
        if best is None or pk > best[1]:
            best = (nid2, pk, c)
    nid2, pk, c = best
    idx = np.where(c > thr * pk)[0]; peaks = []
    for i in idx:  # one peak per cluster
        if not peaks or i - peaks[-1] > nfft:
            peaks.append(i)
        elif c[i] > c[peaks[-1]]:
            peaks[-1] = i
    sp = np.diff(peaks) / rate * 1e3
    return {"rat": rat, "nfft": nfft, "nid2": nid2, "peak_norm_corr": round(pk, 3),
            "median_norm_corr": round(float(np.median(c)), 5), "peak_to_median_db": round(float(10 * np.log10(pk / np.median(c))), 1),
            "peaks": len(peaks), "median_spacing_ms": round(float(np.median(sp)), 3) if len(sp) else None}, c, peaks


def run_nrsib1(base, decoder, rate, freq, win_s=0.1, step_s=0.02, usable_hz=9.36e6, band=78):
    d = base + "_nrsib1"; os.makedirs(d, exist_ok=True)
    dpath, mpath = v.sigmf_paths(base)
    link = os.path.join(d, "capture.sigmf-data")
    if not os.path.exists(link):
        os.symlink(os.path.abspath(dpath), link)
    json.dump({"global": {"core:datatype": "cf32_le", "core:sample_rate": rate, "core:version": "1.2.0"}, "captures": [{"core:frequency": freq, "core:sample_start": 0}], "annotations": []},
              open(os.path.join(d, "capture.sigmf-meta"), "w"))
    n = os.path.getsize(dpath) // 8; W, S = int(win_s * rate), int(step_s * rate); summ = {"windows": 0, "sync": 0, "mib": 0, "sib1": 0, "events": []}
    for start in range(0, n - W + 1, S):
        cfg = {"release": "R18", "band": band, "duplex": "Tdd", "sample_rate_hz": rate, "center_hz": freq, "usable_hz": usable_hz,
               "minimum_channel_bandwidth_hz": 10e6, "spacing": "Khz30", "ssb_case": "C", "gscn": None, "shared_spectrum": False,
               "ntn": False, "minimum_quality_db": 10.0}
        json.dump({"config": cfg, "from": start, "to": start + W}, open(os.path.join(d, "expect.json"), "w"))
        out = subprocess.run([decoder, d], capture_output=True, text=True, timeout=120).stdout
        dec, pos, ev = json.JSONDecoder(), 0, []
        while pos < len(out):
            while pos < len(out) and out[pos].isspace():
                pos += 1
            if pos >= len(out):
                break
            o, pos = dec.raw_decode(out, pos); ev.append(o)
        kinds = [list(e.keys())[0] for e in ev]
        summ["windows"] += 1; summ["sync"] += "Sync" in kinds; summ["mib"] += "Mib" in kinds; summ["sib1"] += "Sib1" in kinds
        summ["events"].append({"from": start, "kinds": kinds, "first": ev[:3], "sib1": [e for e in ev if "Sib1" in e][:1]})
    json.dump(summ, open(base + "_nrsib1.json", "w"), indent=1)
    return {k: summ[k] for k in ("windows", "sync", "mib", "sib1")}, summ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["capture", "check", "nrsib1"]); p.add_argument("base", help="SigMF base name")
    p.add_argument("--rat", choices=["nr", "lte"], default="nr"); p.add_argument("--rate", type=float, default=15.36e6)
    p.add_argument("--freq", type=float, default=3404.64e6); p.add_argument("--gain", type=float, default=40)
    p.add_argument("--secs", type=float, default=0.6); p.add_argument("--scs", type=float, default=30e3)
    p.add_argument("--ssb-offset", type=float, default=0.0, help="SSB center minus RX center (Hz)")
    p.add_argument("--check-secs", type=float, default=0.2); p.add_argument("--decoder", help="nr-sib1 examples/decode binary")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    if a.mode == "capture":
        from iq_recorder import record
        _, st, _ = record(a.rate, a.freq, a.gain, a.secs, a.base, f"dl_capture {a.rat}")
        r, c, pk = pss_check(np.fromfile(v.sigmf_paths(a.base)[0], np.complex64)[:int(a.rate * a.check_secs)].astype(complex), a.rate, a.rat, a.scs, a.ssb_offset)
        print(json.dumps({"capture": st, "pss": r}), flush=True); v.hard_exit(0)
    x, rate, freq, _ = v.read_iq(a.base)
    if a.mode == "check":
        r, c, pk = pss_check(x[:int(rate * a.check_secs)].astype(complex), rate, a.rat, a.scs, a.ssb_offset)
        json.dump(r, open(a.base + "_pss.json", "w"), indent=1); print(json.dumps(r), flush=True)
        if a.plot:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(np.arange(len(c)) / rate * 1e3, c, lw=0.6); ax.set_xlabel("time in record (ms)")
            ax.set_ylabel(f"normalized PSS correlation (NID2 {r['nid2']})"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(a.base + "_pss.png", dpi=150)
    else:
        s, _ = run_nrsib1(a.base, a.decoder, rate, freq); print(json.dumps(s), flush=True)


if __name__ == "__main__":
    main()
