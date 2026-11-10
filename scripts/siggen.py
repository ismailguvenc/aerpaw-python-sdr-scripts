#!/usr/bin/env python3
"""siggen: continuous test-signal generator for the AERPAW V-USRP.

Kinds: tone, multitone (Schroeder phases), chirp (linear FM, sawtooth), pn (1023-chip m-sequence), zc (Zadoff-Chu),
noise (band-limited Gaussian), ofdm (OFDM-like, QPSK on used subcarriers, CP), qpsk / 16qam (RRC single carrier,
known symbols from a seed: meters.py rebuilds them), bursts (noise bursts in K sub-channels with a duty cycle per
channel, zero fill between bursts), zeros (registers the TX only: CHEM then adds its noise at the RX).
Rule: one waveform period is repeated in one continuous stream (no burst tags, zero fill), TX center in an allowed band (rule 5),
1-23.04 MS/s, absolute gain, RMS level set in dBFS (keep the RX below the -16 dBFS clip).
"""
import argparse, json, sys, time
import numpy as np

KINDS = ("tone", "multitone", "chirp", "pn", "zc", "noise", "ofdm", "qpsk", "16qam", "bursts", "zeros")


def mseq1023():
    reg, s = 0x3FF, []
    for _ in range(1023):
        b = ((reg >> 9) ^ (reg >> 6)) & 1; s.append(1.0 if reg & 1 else -1.0); reg = ((reg << 1) | b) & 0x3FF
    return np.array(s, np.complex64)


def zadoff_chu(n=1021, u=7):
    k = np.arange(n)
    return np.exp(-1j * np.pi * u * k * (k + (n % 2)) / n).astype(np.complex64)


def rrc(sps, beta=0.35, span=8):
    t = np.arange(-span * sps, span * sps + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


def qam_symbols(mod, nsym, seed):
    rng = np.random.default_rng(seed)
    if mod == "qpsk":
        b = rng.integers(0, 4, nsym); return (np.exp(1j * (np.pi / 4 + np.pi / 2 * b))).astype(np.complex64)
    lv = np.array([-3, -1, 1, 3]); i = rng.integers(0, 4, nsym); q = rng.integers(0, 4, nsym)
    return ((lv[i] + 1j * lv[q]) / np.sqrt(10)).astype(np.complex64)


def qam_wave(mod, nsym=2048, sps=4, beta=0.35, seed=1):
    """Periodic RRC waveform: circular filtering of a periodic symbol sequence, so the period repeats cleanly."""
    s = qam_symbols(mod, nsym, seed)
    up = np.zeros(nsym * sps, np.complex64); up[::sps] = s
    h = rrc(sps, beta); H = np.fft.fft(h, len(up))
    x = np.fft.ifft(np.fft.fft(up) * H)
    x = np.roll(x, -(len(h) // 2))
    return x.astype(np.complex64), s


def make(kind, rate, args):
    rng = np.random.default_rng(args.seed)
    n = int(rate * args.period_ms / 1e3)
    t = np.arange(n) / rate
    if kind == "tone":
        f = round(args.f0 * args.period_ms / 1e3) / (args.period_ms / 1e3)  # integer cycles per period
        return np.exp(2j * np.pi * f * t)
    if kind == "multitone":
        K = args.ntones; fs = (np.arange(K) - (K - 1) / 2) * args.spacing
        ph = np.pi * np.arange(K) * (np.arange(K) - 1) / K  # Schroeder phases (low PAPR)
        return np.sum([np.exp(1j * (2 * np.pi * f * t + p)) for f, p in zip(fs, ph)], axis=0)
    if kind == "chirp":
        T = n / rate; k = args.bw / T
        return np.exp(1j * np.pi * k * (t - T / 2) ** 2)
    if kind == "pn":
        return mseq1023()
    if kind == "zc":
        return zadoff_chu(args.zc_len, args.zc_root)
    if kind == "noise":
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n); W = np.fft.fft(w); f = np.fft.fftfreq(n, 1 / rate)
        W[np.abs(f) > args.bw / 2] = 0; return np.fft.ifft(W)
    if kind == "ofdm":
        N, cp = args.nfft, args.nfft // 8; used = int(args.nfft * args.bw / rate) // 2
        sym = []
        for _ in range(max(1, n // (N + cp))):
            X = np.zeros(N, complex); k = np.r_[1:used + 1, -used:0]
            X[k] = np.exp(1j * np.pi / 4 * (2 * rng.integers(0, 4, len(k)) + 1)); s = np.fft.ifft(X)
            sym.append(np.r_[s[-cp:], s])
        return np.concatenate(sym)
    if kind in ("qpsk", "16qam"):
        return qam_wave(kind, args.nsym, args.sps, args.beta, args.seed)[0]
    if kind == "bursts":
        offs = [float(v) for v in args.chan_offsets.split(",")]; duty = [float(v) for v in args.duty.split(",")]
        x = np.zeros(n, complex); f = np.fft.fftfreq(n, 1 / rate)
        for o, d in zip(offs, duty):
            w = rng.standard_normal(n) + 1j * rng.standard_normal(n); W = np.fft.fft(w)
            W[np.abs(f - o) > args.chan_bw / 2 * 0.8] = 0; b = np.fft.ifft(W); b /= np.sqrt(np.mean(np.abs(b) ** 2))
            gate = np.zeros(n); gate[:int(d * n)] = 1  # on for the first duty*period, zeros after (zero fill)
            x += b * gate
        return x
    return np.zeros(4096, complex)


def build(args):
    x = make(args.kind, args.rate, args)
    if args.kind == "zeros":
        return x.astype(np.complex64)
    from vusrp_io import scale_to_dbfs
    x = scale_to_dbfs(x, args.rms_dbfs)  # time-average RMS of the full period
    pk = float(np.max(np.abs(x)))
    if pk > 0.99:
        x = (x / pk * 0.99).astype(np.complex64)
    return x


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=KINDS)
    p.add_argument("--rate", type=float, default=2e6); p.add_argument("--freq", type=float, default=3.4e9)
    p.add_argument("--gain", type=float, default=60); p.add_argument("--rms-dbfs", type=float, default=-18)
    p.add_argument("--secs", type=float, default=60, help="TX duration (s)")
    p.add_argument("--period-ms", type=float, default=10); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--f0", type=float, default=100e3, help="tone frequency (Hz)")
    p.add_argument("--ntones", type=int, default=8); p.add_argument("--spacing", type=float, default=100e3)
    p.add_argument("--bw", type=float, default=1e6, help="chirp/noise/ofdm bandwidth (Hz)")
    p.add_argument("--nfft", type=int, default=256); p.add_argument("--zc-len", type=int, default=1021); p.add_argument("--zc-root", type=int, default=7)
    p.add_argument("--nsym", type=int, default=2048); p.add_argument("--sps", type=int, default=4); p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--chan-offsets", default="-750e3,-250e3,250e3,750e3"); p.add_argument("--chan-bw", type=float, default=250e3)
    p.add_argument("--duty", default="1.0,0.5,0.25,0.0", help="bursts: on-fraction per channel")
    p.add_argument("--gain-sched", default="", help="'t:gain,t:gain' TX gain changes at run time (s)")
    p.add_argument("--save", help="write one period as complex64 (.c64) and stop (no radio)")
    return p


def main():
    a = parser().parse_args()
    import vusrp_io as v
    v.check_rate(a.rate); v.check_tx_freq(a.freq, a.rate)
    x = build(a)
    info = {"kind": a.kind, "rate": a.rate, "freq": a.freq, "period_samples": len(x), "rms_dbfs": round(v.dbfs(x), 2) if a.kind != "zeros" else None,
            "peak_dbfs": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 2), "papr_db": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)) - v.dbfs(x), 2) if a.kind != "zeros" else None}
    print(json.dumps(info), flush=True)
    if a.save:
        x.tofile(a.save); json.dump(vars(a), open(a.save + ".json", "w")); return
    tx = v.TX(a.rate, a.freq, a.gain).start(x, a.secs)
    sched = sorted((float(t), float(g)) for t, g in (p.split(":") for p in a.gain_sched.split(",") if p))
    t0 = time.time()
    while tx.th.is_alive():
        while sched and time.time() - t0 >= sched[0][0]:
            tx.set_gain(sched[0][1]); print(f"{time.time()-t0:.1f} s: TX gain {sched[0][1]}", flush=True); sched.pop(0)
        time.sleep(0.1)
    print(json.dumps({"sent_samples": tx.sent, "seconds": round(time.time() - t0, 1)}), flush=True)
    v.hard_exit(0)


if __name__ == "__main__":
    main()
