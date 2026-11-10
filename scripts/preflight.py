#!/usr/bin/env python3
"""preflight: V-USRP self-test before a campaign (environment, device, CHEM, TX/RX tone loop, level and rate).

check : environment file present and sourced, CHEM ports 5000/6000, V-USRP found (uhd_find_devices), device opens,
        rates 1/2/23.04 MS/s accepted, the code rules refuse 0.5 MS/s and a 3.5 GHz TX (no radio call for those).
tx    : continuous tone (--f0) on this node for --secs (run first, on the TX node).
rx    : records --secs on this node and checks: CHEM lists a link to this node at --freq, tone found within +-50 Hz
        of f0, tone-to-floor ratio >= --min-db, sample count per wall-clock second within 2 % of the rate, no
        zero-filled samples, no overflow. Prints PASS/FAIL per item.
Rule: start the TX first; TX center in an allowed band (rule 5); same center frequency and rate on both nodes; skip the first second;
wait 20 s after a radio stops before the next radio starts on that node (CHEM node timeout).
"""
import argparse, json, os, subprocess, time
import numpy as np
import vusrp_io as v


def check():
    r = {"env_file": os.path.exists(v.ENV_FILE), "env_sourced": v.env_ok(), "node": v.node_meta()}
    host = os.environ.get("AP_EXPENV_CHEMVM_XE", "")  # set by the AERPAW environment file
    r["chem_ports"] = {pt: v.port_open(host, pt) for pt in v.CHEM_PORTS}
    try:
        out = subprocess.run(["uhd_find_devices"], capture_output=True, text=True, timeout=30).stdout
        r["find_devices"] = [l.strip() for l in out.splitlines() if "serial" in l or "type" in l or "name" in l][:4]
    except Exception as e:
        r["find_devices"] = f"error {e}"
    for bad in ((lambda: v.check_rate(0.5e6)), (lambda: v.check_tx_freq(3.5e9))):
        try:
            bad(); r.setdefault("rule_refusals", []).append("NOT refused")
        except v.RuleError as e:
            r.setdefault("rule_refusals", []).append(str(e)[:60])
    try:
        import uhd
        u = uhd.usrp.MultiUSRP("")
        r["pp_string"] = u.get_pp_string().strip().splitlines()[:6]
        acc = {}
        for rate in (1e6, 2e6, 23.04e6):
            u.set_rx_rate(rate, 0); acc[f"{rate/1e6:g}"] = round(u.get_rx_rate(0) / 1e6, 4)
        r["rx_rates_msps"] = acc; r["master_clock_hz"] = u.get_master_clock_rate()
    except Exception as e:
        r["device_error"] = str(e)[:200]
    r["pass"] = bool(r["env_sourced"] and all(r["chem_ports"].values()) and "rx_rates_msps" in r)
    return r


def rx_test(a):
    from iq_recorder import record
    meta, st, x = record(a.rate, a.freq, a.gain, a.secs, a.out, "preflight", None, v.SKIP_S)
    xd = x.astype(np.complex128); n = 1 << 16; f = np.fft.fftfreq(n, 1 / a.rate)
    S = np.mean(np.abs(np.fft.fft(xd[:(len(xd) // n) * n].reshape(-1, n) * np.hanning(n), axis=1)) ** 2, axis=0)
    i = int(np.argmax(S)); tone_db = float(10 * np.log10(S[i] / np.median(S)))
    rows = v.load_chem_csv(a.out + "_chem.csv") if os.path.exists(a.out + "_chem.csv") else []
    me = os.environ.get("AP_EXPENV_THIS_CONTAINER_NAME", "?")
    links = sorted({(r["src"][-7:-5], r["dest"][-7:-5], round(r["snr"], 1)) for r in rows if me in str(r["dest"])})
    res = {"chem_link_to_me": (bool(links), links[:4]),
           "tone_freq_hz": (abs(f[i] - a.f0) <= 50, round(float(f[i]), 1)),
           "tone_to_floor_db": (tone_db >= a.min_db, round(tone_db, 1)),
           "rate_vs_wallclock_msps": (abs(len(x) / st["wall_s"] / a.rate - 1) < 0.02, round(len(x) / st["wall_s"] / 1e6, 4)),
           "zero_frac": (st["zero_frac"] < 1e-3, st["zero_frac"]),
           "overflows": (st["overflows"] == 0, st["overflows"]),
           "level_dbfs": (st["max_1ms_dbfs"] < v.CLIP_DBFS, st["rms_dbfs"])}
    out = {k: {"pass": bool(p), "value": val} for k, (p, val) in res.items()}
    out["PASS"] = all(o["pass"] for o in out.values())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("role", choices=["check", "tx", "rx"]); p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--freq", type=float, default=3.4e9); p.add_argument("--gain", type=float, default=None)
    p.add_argument("--f0", type=float, default=100e3); p.add_argument("--rms-dbfs", type=float, default=-18)
    p.add_argument("--secs", type=float, default=5); p.add_argument("--min-db", type=float, default=20)
    p.add_argument("--out", default="preflight_rx")
    a = p.parse_args()
    if a.role == "check":
        print(json.dumps(check(), default=str), flush=True); v.hard_exit(0)
    v.require_env()
    if a.role == "tx":
        n = int(a.rate / 1e3) * 10; x = np.exp(2j * np.pi * a.f0 * np.arange(n) / a.rate)
        tx = v.TX(a.rate, a.freq, a.gain or 60).start(v.scale_to_dbfs(x, a.rms_dbfs), a.secs); tx.wait()
        print(json.dumps({"tx_sent": tx.sent}), flush=True); v.hard_exit(0)
    a.gain = a.gain or 40
    print(json.dumps(rx_test(a)), flush=True); v.hard_exit(0)


if __name__ == "__main__":
    main()
