#!/usr/bin/env python3
"""mat2sigmf: convert AERPAW UHD2 (get_IQ.py) .mat snapshots and raw complex64 files to SigMF.

mat : files results_<LO>_<fs>_<YYYY_MM_DD_HH_MM_SS_mmm>.mat (variable IQ, time_stamp). The snapshots of one LO/rate
      become one cf32_le SigMF record with one capture segment per snapshot (sample_start, frequency, datetime from
      the file name, America/New_York) and one annotation per snapshot with its RMS level.
raw : one raw complex64 file (iqcap.py, fxtest.py, GNU Radio file sink) with --rate and --freq.
Optional: --chem <csv> adds the CHEM rows of the record time span (chem_logger.py or iq_recorder) as annotations.
Each output is validated with sigmf-python when it is installed in the private _vendor folder.
Rule (twin note): UHD2 snapshots at an LO other than the TX frequency hold no CHEM signal (L11); the converter keeps
the LO per capture segment so a reader can separate them.
"""
import argparse, datetime, glob, json, os, re
import numpy as np
import vusrp_io as v

NAME = re.compile(r"results_(\d+)_(\d+)_(\d{4}_\d\d_\d\d_\d\d_\d\d_\d\d_\d{3})\.mat$")


def ny_to_utc(s):
    from zoneinfo import ZoneInfo
    t = datetime.datetime.strptime(s, "%Y_%m_%d_%H_%M_%S_%f").replace(tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(datetime.timezone.utc)


def convert_mat(files, out_dir, chem=None):
    import scipy.io
    groups = {}
    for f in sorted(files):
        m = NAME.search(os.path.basename(f))
        if m:
            groups.setdefault((int(m.group(1)), int(m.group(2))), []).append((m.group(3), f))
    res = []
    for (lo, fs), items in sorted(groups.items()):
        xs, caps, ann, pos = [], [], [], 0
        for ts, f in sorted(items):
            x = scipy.io.loadmat(f)["IQ"].ravel().astype(np.complex64); t = ny_to_utc(ts)
            caps.append({"core:sample_start": pos, "core:frequency": float(lo), "core:datetime": t.strftime("%Y-%m-%dT%H:%M:%S.%fZ")})
            ann.append({"core:sample_start": pos, "core:sample_count": len(x), "core:label": "uhd2_snapshot",
                        "core:comment": os.path.basename(f), "aerpaw:rms_dbfs": round(v.dbfs(x), 2)})
            xs.append(x); pos += len(x)
        if chem:
            rows = v.load_chem_csv(chem)
            t0 = ny_to_utc(items[0][0]).timestamp(); t1 = ny_to_utc(items[-1][0]).timestamp()
            for r in rows:
                if t0 <= r.get("t", 0) <= t1:
                    ann.append({"core:sample_start": 0, "core:sample_count": 0, "core:label": "chem", "aerpaw:t_unix": r["t"],
                                "aerpaw:chem_snr": r.get("snr"), "aerpaw:chem_distance": r.get("distance")})
        base = os.path.join(out_dir, f"uhd2_{lo}_{fs}")
        meta = v.write_sigmf(base, np.concatenate(xs), fs, lo, f"AERPAW UHD2 get_IQ.py snapshots LO {lo/1e6:.2f} MHz",
                             extra_global={"aerpaw:source": "UHD2 sample get_IQ.py (.mat)", "aerpaw:snapshots": len(xs)},
                             annotations=ann, captures=caps)
        lv = [a["aerpaw:rms_dbfs"] for a in ann if a["core:label"] == "uhd2_snapshot"]
        res.append({"lo_hz": lo, "rate": fs, "snapshots": len(xs), "samples": pos, "median_rms_dbfs": round(float(np.median(lv)), 2),
                    "max_rms_dbfs": round(float(np.max(lv)), 2), "meta": meta, "validate": v.sigmf_validate(meta)})
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["mat", "raw"]); p.add_argument("inputs", nargs="+", help="files or globs")
    p.add_argument("--out-dir", default="."); p.add_argument("--rate", type=float); p.add_argument("--freq", type=float)
    p.add_argument("--chem", help="CHEM CSV (chem_logger / iq_recorder)")
    a = p.parse_args()
    files = [f for g in a.inputs for f in (glob.glob(g) or [g])]
    os.makedirs(a.out_dir, exist_ok=True)
    if a.mode == "mat":
        res = convert_mat(files, a.out_dir, a.chem)
    else:
        res = []
        for f in files:
            x = np.fromfile(f, np.complex64); base = os.path.join(a.out_dir, os.path.splitext(os.path.basename(f))[0])
            meta = v.write_sigmf(base, x, a.rate, a.freq, f"converted from {os.path.basename(f)}",
                                 t0=os.path.getmtime(f) - len(x) / a.rate, extra_global={"aerpaw:source": f})
            res.append({"file": f, "samples": len(x), "rms_dbfs": round(v.dbfs(x), 2), "meta": meta, "validate": v.sigmf_validate(meta)})
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
