#!/usr/bin/env python3
"""meters: offline SNR, EVM (QPSK/16QAM) and PAPR-CCDF meters for an I/Q capture (SigMF or raw complex64).

SNR methods: "da" = data-aided (least-squares fit of the known periodic TX waveform, full sample bandwidth);
"noise" = signal-plus-noise power against a noise-only capture; "spectral" = in-band power against the out-of-band
noise density. EVM: RRC matched filter on the aligned record, symbols from the siggen seed, per-period LS gain.
Rule: drop the first second (done by the recorder), compare only records at one rate and center frequency, and
report the share of samples near the -16 dBFS clip (CHEM clips the RX level there).
"""
import argparse, json, sys
import numpy as np
import vusrp_io as v


def align(x, w):
    """Lag of the periodic reference w in x (circular correlation over the first period)."""
    P = len(w); c = np.fft.ifft(np.fft.fft(x[:P]) * np.conj(np.fft.fft(w)))
    lag = int(np.argmax(np.abs(c))); return lag, float(np.abs(c[lag]) / (np.mean(np.abs(c)) + 1e-30))


def snr_da(x, w, block=None):
    """Data-aided SNR over the full sample band: y = h*ref + n with h fitted per block (slow phase drift allowed)."""
    lag, pk = align(x, w); P = len(w); block = block or P
    ref = w[(np.arange(len(x)) - lag) % P]
    S = N = 0.0
    for i in range(0, len(x) - block + 1, block):
        y, r = x[i:i + block], ref[i:i + block]
        h = np.vdot(r, y) / np.vdot(r, r); e = y - h * r
        S += np.abs(h) ** 2 * np.mean(np.abs(r) ** 2); N += np.mean(np.abs(e) ** 2)
    return {"snr_da_db": round(float(10 * np.log10(S / N)), 2), "lag": lag, "corr_peak_to_mean": round(pk, 1)}


def snr_noise(x, noise):
    ps, pn = np.mean(np.abs(x) ** 2), np.mean(np.abs(noise) ** 2)
    return {"snr_noise_db": round(float(10 * np.log10(max(ps - pn, 1e-30) / pn)), 2), "noise_dbfs": round(float(10 * np.log10(pn)), 2)}


def snr_spectral(x, rate, sig_bw, nfft=4096):
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1 / rate))
    K = len(x) // nfft; X = x[:K * nfft].reshape(K, nfft) * np.hanning(nfft)
    S = np.fft.fftshift(np.mean(np.abs(np.fft.fft(X, axis=1)) ** 2, axis=0))
    inb = np.abs(f) <= sig_bw / 2; oob = (np.abs(f) > sig_bw / 2 * 1.15) & (np.abs(f) < 0.45 * rate)
    n0 = np.mean(S[oob]); sig = np.sum(S[inb]) - n0 * inb.sum()
    return {"snr_spectral_inband_db": round(float(10 * np.log10(max(sig, 1e-30) / (n0 * inb.sum()))), 2),
            "snr_spectral_fullband_db": round(float(10 * np.log10(max(sig, 1e-30) / (n0 * nfft))), 2)}


def evm(x, mod, nsym=2048, sps=4, beta=0.35, seed=1):
    from siggen import qam_wave, rrc
    w, s = qam_wave(mod, nsym, sps, beta, seed); P = len(w)
    lag, _ = align(x, w); h = rrc(sps, beta); H = np.fft.fft(h, P)
    errs, pts = [], []
    for i in range(lag, len(x) - P + 1, P):
        z = np.roll(np.fft.ifft(np.fft.fft(x[i:i + P]) * H), -(len(h) // 2))[::sps]
        g = np.vdot(s, z) / np.vdot(s, s); zn = z / g
        errs.append(np.mean(np.abs(zn - s) ** 2)); pts.append(zn)
    e = float(np.sqrt(np.mean(errs) / np.mean(np.abs(s) ** 2)))
    return {"mod": mod, "periods": len(errs), "evm_pct": round(100 * e, 2), "evm_db": round(20 * np.log10(e), 2),
            "snr_from_evm_db": round(-20 * np.log10(e), 2)}, (np.concatenate(pts) if pts else np.array([])), s


def papr(x, levels=np.arange(0, 13, 0.25)):
    p = np.abs(x) ** 2; r = p / np.mean(p)
    ccdf = np.array([np.mean(10 * np.log10(r + 1e-20) > L) for L in levels])
    i = np.where(ccdf <= 1e-3)[0]
    return {"papr_max_db": round(float(10 * np.log10(r.max())), 2),
            "papr_at_1e-3_db": round(float(levels[i[0]]), 2) if len(i) else None,
            "clip_frac": round(float(np.mean(np.abs(x) > 10 ** (v.CLIP_DBFS / 20) * 0.98)), 6)}, levels, ccdf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file"); p.add_argument("--rate", type=float, help="raw files only")
    p.add_argument("--ref", help="one TX period (.c64 from siggen --save) for the data-aided SNR")
    p.add_argument("--noise", help="noise-only capture (TX zeros) for the power SNR")
    p.add_argument("--sig-bw", type=float, help="signal bandwidth (Hz) for the spectral SNR")
    p.add_argument("--mod", choices=["qpsk", "16qam"], help="EVM of a siggen qpsk/16qam record")
    p.add_argument("--nsym", type=int, default=2048); p.add_argument("--sps", type=int, default=4); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-secs", type=float, default=3.0); p.add_argument("--plot", help="PNG: constellation + CCDF")
    a = p.parse_args()
    x, rate, freq, _ = v.read_iq(a.file, a.rate); x = x[:int(rate * a.max_secs)].astype(np.complex128)
    r = {"file": a.file, "samples": len(x), "rms_dbfs": round(v.dbfs(x), 2)}
    if a.ref:
        r.update(snr_da(x, np.fromfile(a.ref, np.complex64).astype(np.complex128)))
    if a.noise:
        r.update(snr_noise(x, v.read_iq(a.noise, a.rate)[0][:len(x)].astype(np.complex128)))
    if a.sig_bw:
        r.update(snr_spectral(x, rate, a.sig_bw))
    pts = None
    if a.mod:
        e, pts, s = evm(x, a.mod, a.nsym, a.sps, 0.35, a.seed); r.update(e)
    pr, lv, cc = papr(x); r.update(pr)
    print(json.dumps(r), flush=True)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9})
        fig, ax = plt.subplots(1, 2 if pts is not None else 1, figsize=(6, 3), squeeze=False); ax = ax[0]
        ax[-1].semilogy(lv, np.maximum(cc, 1e-6)); ax[-1].set_xlabel("PAPR threshold (dB)"); ax[-1].set_ylabel("CCDF"); ax[-1].grid(alpha=.3)
        if pts is not None:
            ax[0].plot(pts.real[:4000], pts.imag[:4000], ".", ms=1.5); ax[0].set_aspect("equal"); ax[0].set_title(f"{a.mod} EVM {r['evm_pct']} %")
        fig.tight_layout(); fig.savefig(a.plot, dpi=150)


if __name__ == "__main__":
    main()
