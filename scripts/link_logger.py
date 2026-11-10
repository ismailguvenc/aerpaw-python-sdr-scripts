#!/usr/bin/env python3
"""link_logger: one time-aligned CSV of CHEM link state, ping RTT and iperf3 throughput during a link test.

Threads: CHEM linkinfo (pychem, every --chem-period s), ping to --target (one row per reply or loss), iperf3 client
to --target (one row per 1-s interval, --forceflush). Output: <out>.csv (long format: unix time, source, metric,
value), <out>_1s.csv (one row per second: CHEM SNR/distance, ping RTT, throughput), <out>.json (summary), PNG.
Rule: no radio access (it only reads CHEM and runs network tools); the M1/M3 clocks differ by about 25 ms (L26),
so the logger runs on one node and timestamps everything there. Start it after the UE has an IP address.
"""
import argparse, csv, json, re, subprocess, threading, time
import numpy as np
import vusrp_io as v

ROWS = []; LOCK = threading.Lock()


def add(src, metric, value, t=None):
    with LOCK:
        ROWS.append((t or time.time(), src, metric, value))


def chem_thread(sel, period, stop):
    try:
        c, _ = v.chem_client()
    except Exception as e:
        add("chem", "error", str(e)); return
    while not stop.is_set():
        try:
            for L in v.chem_links(c, sel):
                for k in ("snr", "snrMeasuredDb", "distance", "plDb", "dopplerHz"):
                    if L.get(k) is not None:
                        add("chem", f"{L['src'][-7:-5]}>{L['dest'][-7:-5]}:{k}", float(L[k]))
        except Exception as e:
            add("chem", "error", str(e)[:80])
        stop.wait(period)


def ping_thread(target, interval, secs, iface):
    cmd = ["ping", "-D", "-O", "-i", str(interval), "-w", str(int(secs)), target] + (["-I", iface] if iface else [])
    for line in subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True).stdout:
        m = re.search(r"\[(\d+\.\d+)\].*icmp_seq=(\d+).*time=([\d.]+)", line)
        if m:
            add("ping", "rtt_ms", float(m.group(3)), float(m.group(1)))
        elif "no answer" in line:
            m = re.search(r"\[(\d+\.\d+)\]", line); add("ping", "lost", 1, float(m.group(1)) if m else None)


def iperf_thread(target, secs, extra, delay, raw):
    time.sleep(delay)
    cmd = ["iperf3", "-c", target, "-t", str(int(secs)), "-i", "1", "--forceflush"] + extra.split()
    with open(raw, "w") as fr:
        for line in subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout:
            fr.write(line)
            m = re.search(r"\]\s+([\d.]+)-([\d.]+)\s+sec\s+[\d.]+ \w*Bytes\s+([\d.]+) ([KMG]?)bits/sec", line)
            if m:
                mbps = float(m.group(3)) * {"K": 1e-3, "M": 1, "G": 1e3}.get(m.group(4), 1e-6)
                if float(m.group(2)) - float(m.group(1)) <= 1.5:
                    add("iperf3", "mbps", mbps)  # 1-s interval, zero intervals included
                elif "receiver" in line:
                    add("iperf3", "receiver_mbps", mbps)
            elif "error" in line.lower():
                add("iperf3", "error", line.strip()[:100])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", required=True, help="UE IP (iperf3 server must run there)")
    p.add_argument("--secs", type=float, default=30); p.add_argument("--out", default="linklog")
    p.add_argument("--link", default=None, help="CHEM link filter, for example M1>M3"); p.add_argument("--chem-period", type=float, default=1.0)
    p.add_argument("--ping-interval", type=float, default=0.5); p.add_argument("--iface", default=None)
    p.add_argument("--iperf-secs", type=float, default=20); p.add_argument("--iperf-args", default="", help="for example '-R -u -b 4M'")
    p.add_argument("--iperf-delay", type=float, default=5); p.add_argument("--no-iperf", action="store_true"); p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    stop = threading.Event(); t0 = time.time()
    th = [threading.Thread(target=chem_thread, args=(a.link, a.chem_period, stop), daemon=True),
          threading.Thread(target=ping_thread, args=(a.target, a.ping_interval, a.secs, a.iface), daemon=True)]
    if not a.no_iperf:
        th.append(threading.Thread(target=iperf_thread, args=(a.target, a.iperf_secs, a.iperf_args, a.iperf_delay, a.out + "_iperf.txt"), daemon=True))
    [t.start() for t in th]
    while time.time() - t0 < a.secs + 1:
        time.sleep(0.5)
    stop.set(); time.sleep(1.0)
    rows = sorted(ROWS)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_unix", "source", "metric", "value"]); [w.writerow(r) for r in rows]
    sec = {}
    for t, s, m, val in rows:
        if isinstance(val, (int, float)):
            sec.setdefault(int(t - t0), {}).setdefault(f"{s}:{m}", []).append(val)
    cols = sorted({k for d in sec.values() for k in d})
    with open(a.out + "_1s.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["t_s"] + cols)
        for k in sorted(sec):
            w.writerow([k] + [round(float(np.mean(sec[k][c])), 3) if c in sec[k] else "" for c in cols])
    def vals(s, m): return [r[3] for r in rows if r[1] == s and r[2].endswith(m) and isinstance(r[3], (int, float))]
    rtt, lost = vals("ping", "rtt_ms"), vals("ping", "lost")
    mb = [r[3] for r in rows if r[1] == "iperf3" and r[2] == "mbps"]
    snr = {r[2]: [] for r in rows if r[1] == "chem" and r[2].endswith(":snr")}
    for r in rows:
        if r[2] in snr:
            snr[r[2]].append(r[3])
    summ = {"secs": a.secs, "rows": len(rows), "ping_replies": len(rtt), "ping_lost": len(lost),
            "ping_rtt_median_ms": round(float(np.median(rtt)), 2) if rtt else None, "ping_rtt_p90_ms": round(float(np.percentile(rtt, 90)), 2) if rtt else None,
            "iperf_intervals": len(mb), "iperf_mean_mbps": round(float(np.mean(mb)), 2) if mb else None,
            "iperf_zero_intervals": int(sum(1 for x in mb if x == 0)), "iperf_receiver_mbps": (vals("iperf3", "receiver_mbps") or [None])[-1],
            "chem_snr_median_db": {k: round(float(np.median(x)), 2) for k, x in snr.items()},
            "errors": [r[3] for r in rows if r[2] == "error"][:3]}
    json.dump(summ, open(a.out + ".json", "w"), indent=1); print(json.dumps(summ), flush=True)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9}); fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
        tr = [r[0] - t0 for r in rows if r[1] == "ping" and r[2] == "rtt_ms"]
        ax[0].plot(tr, rtt, ".", ms=3); ax[0].set_ylabel("ping RTT (ms)"); ax[0].grid(alpha=.3)
        ti = [r[0] - t0 for r in rows if r[1] == "iperf3" and r[2] == "mbps"]
        ax[1].plot(ti, mb, "o-", ms=3, label="iperf3 (Mbit/s)")
        for k, x in snr.items():
            ts = [r[0] - t0 for r in rows if r[2] == k]; ax[1].plot(ts, x, "--", label=f"CHEM SNR {k.split(':')[0]} (dB)")
        ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=150)
    v.hard_exit(0)


if __name__ == "__main__":
    main()
