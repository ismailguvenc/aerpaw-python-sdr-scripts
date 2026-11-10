#!/usr/bin/env python3
"""chem_sweeper: loop over CHEM settings (fixed SNR, frequency offset or CIR) with an automatic capture per point.

Runs on the RX node while a known signal (siggen) runs on the TX node. For each value: apply the setting to the
selected link with pychem, keep the RX stream running for --settle s (the setting acts after about 0.5 s), read the CHEM link
state, record --secs, run meters (data-aided SNR against the siggen period, EVM for qpsk/16qam) and write one
CSV/JSON row. One RX stream serves the whole sweep. At the end the link goes back to --restore.
Rule: one change at a time, a settle time per change (all CHEM commands acted within 0.6 s in the R1/R3 tests of 5 Nov 2026;
3 s is the tested value for a fixed SNR; earlier 7-16 s values came from a control-computer clock 13.3 s behind the nodes), no retune,
CIR taps in samples (L16), skip the first second, continuous RX.
"""
import argparse, csv, json, time
import numpy as np
import vusrp_io as v
import meters

SETTLE = {"noise": 3.0, "fo": 3.0, "cir": 3.0}  # version 1 (4 Nov 2026): 16, 8, 8


def apply(c, pc, param, value, sel):
    out = {}
    for f, ls in (c.get_individual_channels() or {}).items():
        for L in ls:
            s, d = L["src"], L["dest"]
            if sel and not (sel.split(">")[0] in s and sel.split(">")[1] in d):
                continue
            if param == "noise":
                out[f"{s}>{d}"] = c.set_noise_model(float(f), s, d, "FIXED_SNR", float(value)) if value not in ("LINK_BUDGET", "NONE") \
                    else c.set_noise_model(float(f), s, d, value, 0.0)
            elif param == "fo":
                out[f"{s}>{d}"] = c.set_frequency_offset(float(f), s, d, float(value))
            elif param == "cir":
                alias = {"unit": "Single-path (unit)", "echo2": "Two-path echo (3 samples, -6 dB)", "three": "Three-path (0,2,5 samples)", "dense8": "Dense 8-tap (fixed)"}
                taps = [] if value == "clear" else pc.CIR_SCENARIOS[alias.get(value, value)]
                out[f"{s}>{d}"] = c.set_cir(float(f), s, d, 0, 0, taps)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("param", choices=["noise", "fo", "cir"]); p.add_argument("values", help="comma list, for example 0,5,10,LINK_BUDGET")
    p.add_argument("--link", default="M1>M3"); p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=40); p.add_argument("--secs", type=float, default=2.0)
    p.add_argument("--settle", type=float, help="seconds after a change (default 3)")
    p.add_argument("--ref", help="siggen period file (.c64) for the data-aided SNR"); p.add_argument("--mod", choices=["qpsk", "16qam"])
    p.add_argument("--restore", default=None, help="value applied at the end (noise: LINK_BUDGET, fo: 0, cir: clear)")
    p.add_argument("--save-iq", action="store_true"); p.add_argument("--out", default="sweep")
    a = p.parse_args()
    v.require_env(); c, pc = v.chem_client(); settle = a.settle or SETTLE[a.param]
    ref = np.fromfile(a.ref, np.complex64).astype(np.complex128) if a.ref else None
    rx = v.RX(a.rate, a.freq, a.gain); rx.start(); rx.read(int(a.rate), 0.0)  # registration + first second
    rows = []
    for val in a.values.split(","):
        t_set = time.time(); ack = apply(c, pc, a.param, val, a.link)
        x, t0 = rx.read(int(a.rate * a.secs), settle)
        link = (v.chem_links(c, a.link, a.freq / 1e6) or [{}])[0]
        r = {"param": a.param, "value": val, "settle_s": settle, "t_set": round(t_set, 2), "ack": str(list(ack.values())[:1])[:80],
             "chem_snr": link.get("snr"), "chem_snrMeasuredDb": link.get("snrMeasuredDb"), "chem_noiseModel": link.get("noiseModel"),
             "chem_freqOffsetHz": link.get("freqOffsetHz"), "chem_cirTapCount": link.get("cirTapCount"), "rms_dbfs": round(v.dbfs(x), 2)}
        xd = x.astype(np.complex128)
        if ref is not None:
            r.update(meters.snr_da(xd, ref))
        if a.mod:
            r.update(meters.evm(xd, a.mod)[0])
        if a.save_iq:
            v.write_sigmf(f"{a.out}_{a.param}_{val}", x, a.rate, a.freq, f"chem_sweeper {a.param}={val}", t0, {"aerpaw:chem_link": link})
        rows.append(r); print(json.dumps(r), flush=True)
    if a.restore:
        apply(c, pc, a.param, a.restore, a.link); print(json.dumps({"restored": a.restore}), flush=True)
    rx.stop()
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    v.hard_exit(0)


if __name__ == "__main__":
    main()
