#!/usr/bin/env python3
"""iq_recorder: SigMF I/Q recorder with a CHEM link log and node metadata (twin replacement for the UHD2 sample).

Output: <out>.sigmf-data (cf32_le), <out>.sigmf-meta (core fields, node metadata and CHEM link state in the
"aerpaw:" namespace, one annotation per second with the CHEM SNR/distance), <out>_chem.csv (CHEM poll at 4 Hz).
Rule: one continuous RX stream at a fixed frequency and rate (no retune, no finite burst), the first second is
dropped (CHEM 0-m registration link), the record goes to disk (the node /dev/shm holds only 64 MB). Start the TX first.
"""
import argparse, json, os, sys, time
import numpy as np
import vusrp_io as v


def record(rate, freq, gain, secs, out, label="", link=None, skip=v.SKIP_S, chem=True, lag_log=False):
    poll = v.ChemPoller(link, 0.25, freq / 1e6) if chem else None
    if poll:
        poll.start()
    rx = v.RX(rate, freq, gain)
    x, t0 = rx.read(int(rate * secs), skip, lag_log); rx.stop()
    if lag_log:
        rx.save_lag(out + "_lag.npy")
    t1 = time.time()
    rows = []
    if poll:
        time.sleep(0.3); poll.stop(); rows = [r for r in poll.rows if t0 - 0.5 <= r["t"] <= t1 + 0.5]
        poll.rows = rows; poll.save(out + "_chem.csv")
    ann = []
    for s in range(int(secs)):
        rs = [r for r in rows if t0 + s <= r["t"] < t0 + s + 1]
        if rs:
            a = {"core:sample_start": int(s * rate), "core:sample_count": int(rate), "core:label": "chem",
                 "core:generator": "iq_recorder"}
            for k in ("distance", "snr", "snrMeasuredDb", "plDb", "dopplerHz", "freqOffsetHz", "cirTapCount", "noiseModel"):
                vals = [r[k] for r in rs if r.get(k) is not None]
                if vals:
                    a["aerpaw:chem_" + k] = vals[-1] if isinstance(vals[-1], str) else round(float(np.median(vals)), 3)
            ann.append(a)
    blk = x[:int(rate * 1e-3) * (len(x) // int(rate * 1e-3))].reshape(-1, int(rate * 1e-3))
    pblk = 10 * np.log10(np.mean(np.abs(blk) ** 2, axis=1) + 1e-20)
    stats = {"rms_dbfs": round(v.dbfs(x), 2), "max_1ms_dbfs": round(float(pblk.max()), 2), "median_1ms_dbfs": round(float(np.median(pblk)), 2),
             "zero_frac": round(float(np.mean(x == 0)), 5), "clip_frac": round(float(np.mean(np.abs(x) > 10 ** (v.CLIP_DBFS / 20) * 0.98)), 5),
             "overflows": rx.overflows, "timeouts": rx.timeouts, "wall_s": round(t1 - t0, 3)}
    g = {"aerpaw:node": v.node_meta(), "aerpaw:label": label, "aerpaw:rx_gain_db": gain, "aerpaw:skip_s": skip,
         "aerpaw:link": link, "aerpaw:stats": stats, "aerpaw:chem_rows": len(rows)}
    if rows:
        g["aerpaw:chem_first"] = {k: rows[0].get(k) for k in ("src", "dest", "distance", "snr", "plDb", "noiseModel")}
    meta = v.write_sigmf(out, x, rate, freq, f"{label} rx {freq/1e6:.2f} MHz {rate/1e6:.3f} MS/s", t0, g, ann)
    return meta, stats, x


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="output base name (no extension)")
    p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=40); p.add_argument("--secs", type=float, default=5)
    p.add_argument("--skip", type=float, default=v.SKIP_S, help="seconds dropped at the start (default 1)")
    p.add_argument("--label", default=""); p.add_argument("--link", default=None, help="CHEM link filter, for example M1>M3")
    p.add_argument("--no-chem", action="store_true")
    p.add_argument("--lag-log", action="store_true", help="also write <out>_lag.npy (wall time and sample count after each read)")
    a = p.parse_args()
    meta, stats, _ = record(a.rate, a.freq, a.gain, a.secs, a.out, a.label, a.link, a.skip, not a.no_chem, a.lag_log)
    stats["meta"] = meta; stats["validate"] = v.sigmf_validate(meta)
    print(json.dumps(stats), flush=True)
    v.hard_exit(0)


if __name__ == "__main__":
    main()
