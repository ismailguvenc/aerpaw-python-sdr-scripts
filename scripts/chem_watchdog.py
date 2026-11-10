#!/usr/bin/env python3
"""chem_watchdog: checks the CHEM ports (5000 control, 6000 V-USRP config) and runs reset_chem when CHEM stops.

Every --period s it opens TCP connections to both ports. When both checks fail for longer than --grace s, it logs an
incident, optionally kills local radio processes (--kill-pattern), runs --reset-cmd once (default: the image's
reset_chem -v) and measures the time until both ports answer again. Every --drops-every checks it also reads the
CHEM log counters ("TX queue is full", "Drops @"). Output: an event CSV and a JSON summary.
Rule: CHEM is shared by every user of the twin. A full experiment reset (/root/reset.sh) also stops CHEM for about
55 s, so keep --grace above that unless you mean to act; --max-resets limits the actions; --dry-run only logs.
"""
import argparse, csv, json, os, re, subprocess, time
import vusrp_io as v

RESET = "/root/AERPAW-Dev/AHN/E-VM/bin/reset_chem -v"


def drops(c):
    ansi = re.compile(r"\x1b\[[0-9;]*m"); lines = [ansi.sub("", str(x)) for x in c.get_logs(2048)]
    return sum("TX queue is full" in x for x in lines), sum("Drops @" in x for x in lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("AP_EXPENV_CHEMVM_XE"),
                   help="CHEM host (default: $AP_EXPENV_CHEMVM_XE from the AERPAW environment file)")
    p.add_argument("--period", type=float, default=5); p.add_argument("--grace", type=float, default=90)
    p.add_argument("--secs", type=float, default=0, help="run time (0 = until killed)")
    p.add_argument("--reset-cmd", default=RESET); p.add_argument("--max-resets", type=int, default=1)
    p.add_argument("--max-wait", type=float, default=180); p.add_argument("--kill-pattern", default="")
    p.add_argument("--drops-every", type=int, default=0); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default="chem_watchdog")
    a = p.parse_args()
    if not a.host:
        p.error("no CHEM host: source the AERPAW environment file or give --host")
    f = open(a.out + ".csv", "a", newline=""); w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["t_unix", "event", "detail"])
    def ev(e, d=""):
        w.writerow([round(time.time(), 2), e, d]); f.flush(); print(time.strftime("%H:%M:%S"), e, d, flush=True)
    t0 = time.time(); down_since = None; resets = 0; n = 0; incidents = []; c = None
    ev("start", f"host {a.host} grace {a.grace} s dry_run {a.dry_run}")
    while not a.secs or time.time() - t0 < a.secs:
        n += 1; up = [v.port_open(a.host, pt) for pt in v.CHEM_PORTS]
        if all(up):
            if down_since:
                incidents[-1]["up_after_s"] = round(time.time() - down_since, 1); ev("up", f"down for {incidents[-1]['up_after_s']} s"); down_since = None
            if a.drops_every and n % a.drops_every == 0:
                try:
                    c = c or v.chem_client()[0]; q, d = drops(c); ev("drops", f"tx_queue_full {q} drops_lines {d}")
                except Exception as e:
                    ev("drops_error", str(e)[:100])
        else:
            if not down_since:
                down_since = time.time(); incidents.append({"t_down": round(down_since, 2), "ports": up}); ev("down", f"ports 5000/6000 open: {up}")
            elif time.time() - down_since > a.grace and "reset_t" not in incidents[-1]:
                if resets >= a.max_resets or a.dry_run:
                    incidents[-1]["reset_t"] = None; ev("no_action", "dry run" if a.dry_run else "max resets reached")
                else:
                    if a.kill_pattern:
                        subprocess.run(["pkill", "-9", "-f", a.kill_pattern]); ev("killed", a.kill_pattern)
                    resets += 1; incidents[-1]["reset_t"] = round(time.time(), 2); ev("reset", a.reset_cmd)
                    r = subprocess.run(a.reset_cmd, shell=True, capture_output=True, text=True, timeout=a.max_wait)
                    ev("reset_done", (r.stdout + r.stderr).strip().replace("\n", " | ")[-200:])
                    for _ in range(int(a.max_wait / 3)):
                        if all(v.port_open(a.host, pt) for pt in v.CHEM_PORTS):
                            break
                        time.sleep(3)
                    incidents[-1]["up_after_reset_s"] = round(time.time() - incidents[-1]["reset_t"], 1)
        time.sleep(a.period)
    s = {"checks": n, "resets": resets, "incidents": incidents, "secs": round(time.time() - t0, 1)}
    ev("stop", json.dumps(s)); json.dump(s, open(a.out + ".json", "w"), indent=1)


if __name__ == "__main__":
    main()
