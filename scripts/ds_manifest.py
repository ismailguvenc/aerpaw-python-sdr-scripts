#!/usr/bin/env python3
"""ds_manifest: turn the SigMF records of chem_sweeper (--save-iq) into a labeled data set (offline, no radio).

usage: ds_manifest.py <sweep_csv> [<sweep_csv> ...] --mod qpsk,16qam --out manifest.csv [--annotate]

For each row of each sweep CSV (one CHEM point), the tool finds the SigMF record <out>_<param>_<value>.sigmf-meta,
writes one manifest row (file, label, CHEM SNR, measured SNR and EVM, level, start time, samples) and, with
--annotate, adds one annotation that covers the whole record with these labels ("core:label" = modulation,
"aerpaw:chem_snr_db", "aerpaw:snr_da_db", "aerpaw:evm_pct"). A record whose measured SNR differs from the CHEM SNR by
more than --tol dB gets the flag "check" (the CHEM change had not acted, or the TX was off).
"""
import argparse, csv, json, os


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+"); p.add_argument("--mod", required=True, help="one label for each CSV, comma list")
    p.add_argument("--out", default="manifest.csv"); p.add_argument("--annotate", action="store_true")
    p.add_argument("--tol", type=float, default=0.5)
    a = p.parse_args()
    mods = a.mod.split(","); rows = []
    for path, mod in zip(a.csv, mods):
        base = path[:-4]
        for r in csv.DictReader(open(path)):
            meta = f"{base}_{r['param']}_{r['value']}.sigmf-meta"
            if not os.path.exists(meta):
                continue
            m = json.load(open(meta)); g = m["global"]
            n = os.path.getsize(meta[:-5] + "data") // 8 if os.path.exists(meta[:-5] + "data") else None
            chem = float(r["chem_snr"]); da = float(r["snr_da_db"])
            row = {"file": os.path.basename(meta[:-11]), "label": mod, "chem_snr_db": round(chem, 2), "snr_da_db": da,
                   "evm_pct": float(r["evm_pct"]), "rms_dbfs": float(r["rms_dbfs"]), "rate": g["core:sample_rate"],
                   "freq": m["captures"][0]["core:frequency"], "datetime": m["captures"][0]["core:datetime"],
                   "samples": n, "flag": "ok" if abs(da - chem) <= a.tol else "check"}
            rows.append(row)
            if a.annotate:
                ann = {"core:sample_start": 0, "core:label": mod, "core:generator": "ds_manifest",
                       "aerpaw:chem_snr_db": row["chem_snr_db"], "aerpaw:snr_da_db": da, "aerpaw:evm_pct": row["evm_pct"]}
                if n:
                    ann["core:sample_count"] = n
                m["annotations"] = [x for x in m.get("annotations", []) if x.get("core:generator") != "ds_manifest"] + [ann]
                json.dump(m, open(meta, "w"), indent=1)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(json.dumps({"records": len(rows), "labels": sorted(set(r["label"] for r in rows)),
                      "check": sum(r["flag"] == "check" for r in rows), "out": a.out}))


if __name__ == "__main__":
    main()
