"""vusrp_io: shared helper for the AERPAW V-USRP (digital twin, experiment X1532, Nov 2026).

Rules that this module enforces in code (from the X1532 debug report, findings L8-L26):
  * rate 1.0-23.04 MS/s (below 1 MS/s the stream has gaps or no signal; above 23.04 MS/s almost no frames arrive)
  * TX center frequency only in the allowed bands TX_BANDS (6.765-6.795, 13.553-13.567, 26.957-27.283, 40.66-40.70, 433.05-434.79,
    860-880, 902-928 and 3300-3450 MHz). The V-USRP refuses other frequencies: "not allowed by AERPAW rules"
  * no retune while a stream runs (a 500 kHz RX retune gives noise only); set rate/freq/gain once before streaming
  * continuous RX only (start_cont); finite bursts (num_done) return 0 samples without an error
  * continuous TX with zero fill between bursts; one start_of_burst, no end_of_burst until the end
  * the first second of an RX record is dropped (CHEM shows a 0-m link for up to 0.75 s after registration)
  * each send() is far below 400 kB ("Too many fragments" above that); recv() uses exactly max_num_samps
    (the UHD1 sample, which stopped CHEM, printed "Buffer size mismatch 2040 15360")
  * absolute gains, sc16 wire format (sc8 and normalized gains do not work)
  * programs leave with os._exit (the V-USRP threads do not stop and ignore SIGTERM)
Also: SigMF writer/reader (core fields + an "aerpaw:" namespace), raw-file reader, a pychem (CHEM) client and poller.
"""
import csv, datetime, json, os, socket, sys, threading, time
import numpy as np

RATE_MIN, RATE_MAX = 1.0e6, 23.04e6
TX_BANDS = [(6.765e6, 6.795e6), (13.553e6, 13.567e6), (26.957e6, 27.283e6), (40.66e6, 40.70e6),
            (433.05e6, 434.79e6), (860e6, 880e6), (902e6, 928e6), (3300e6, 3450e6)]
SKIP_S = 1.0
SEND_MAX_BYTES = 400_000
CLIP_DBFS = -16.0
CHEM_PORTS = (5000, 6000)
ENV_FILE = "/root/.ap-set-experiment-env.sh"
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")  # private sigmf-python (pip --target)


class RuleError(ValueError):
    """A request that breaks a V-USRP rule."""


def check_rate(rate):
    if not RATE_MIN <= rate <= RATE_MAX:
        raise RuleError(f"rate {rate/1e6:.3f} MS/s outside {RATE_MIN/1e6}-{RATE_MAX/1e6} MS/s (V-USRP rule L12/#48)")
    return rate


def check_tx_freq(freq, rate=None):
    """Refuse a TX center frequency outside all TX_BANDS. With rate, warn when fc +- rate/2 leaves the range."""
    for lo, hi in TX_BANDS:
        if lo <= freq <= hi:
            if rate and (freq - rate / 2 < lo or freq + rate / 2 > hi):
                print(f"[vusrp_io] warning: sample band {(freq - rate / 2) / 1e6:.3f}-{(freq + rate / 2) / 1e6:.3f} MHz "
                      f"leaves the allowed range {lo / 1e6:g}-{hi / 1e6:g} MHz", flush=True)
            return freq
    raise RuleError(f"TX frequency {freq/1e6:.2f} MHz outside the allowed bands (AERPAW rule 5)")


def env_ok():
    """True when the experiment environment is sourced (without it the V-USRP cannot find CHEM)."""
    return bool(os.environ.get("AP_EXPENV_THIS_CONTAINER_NAME")) and bool(os.environ.get("AP_EXPENV_CHEMVM_XE"))


def require_env():
    if not env_ok():
        sys.exit(f"experiment environment missing: run 'source {ENV_FILE}' first (V-USRP: 'No devices found')")


def node_meta():
    keys = ("AP_EXPENV_THIS_CONTAINER_NAME", "AP_EXPENV_THIS_CONTAINER_NODE_TYPE", "AP_EXPENV_THIS_CONTAINER_NODE_VEHICLE",
            "AP_EXPENV_EXP_NUM", "AP_EXPENV_SESSION_ENV", "AP_EXPENV_CHEMVM_XE")
    m = {k.replace("AP_EXPENV_", "").lower(): os.environ.get(k) for k in keys if os.environ.get(k)}
    m["hostname"] = socket.gethostname()
    return m


def hard_exit(code=0):
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(code)


def dbfs(x):
    return float(10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-20))


def scale_to_dbfs(x, rms_dbfs):
    x = np.asarray(x, np.complex64)
    return (x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-30) * 10 ** (rms_dbfs / 20)).astype(np.complex64)


# ---------------------------------------------------------------- radio
def _usrp(rate, freq, gain, direction):
    import uhd
    require_env()
    check_rate(rate)
    if direction == "tx":
        check_tx_freq(freq, rate)
    u = uhd.usrp.MultiUSRP("")
    if direction == "rx":
        u.set_rx_rate(rate, 0); u.set_rx_freq(uhd.types.TuneRequest(freq), 0); u.set_rx_gain(gain, 0)
        got = (u.get_rx_rate(0), u.get_rx_freq(0))
    else:
        u.set_tx_rate(rate, 0); u.set_tx_freq(uhd.types.TuneRequest(freq), 0); u.set_tx_gain(gain, 0)
        got = (u.get_tx_rate(0), u.get_tx_freq(0))
    print(f"[vusrp_io] {direction} rate {got[0]:.0f} S/s freq {got[1]:.0f} Hz gain {gain}", flush=True)
    return uhd, u


class RX:
    """Continuous receiver. Rate, frequency and gain are fixed at open time (no retune while streaming)."""

    def __init__(self, rate, freq, gain=40.0):
        self.rate, self.freq, self.gain = rate, freq, gain
        self.uhd, self.u = _usrp(rate, freq, gain, "rx")
        sa = self.uhd.usrp.StreamArgs("fc32", "sc16"); sa.channels = [0]
        self.st = self.u.get_rx_stream(sa)
        self.spp = self.st.get_max_num_samps()
        self.md = self.uhd.types.RXMetadata()
        self.running = False
        self.overflows = 0; self.timeouts = 0; self.lag_rows = []; self.total = 0

    def set_freq(self, *_):
        raise RuleError("no retune while streaming (L17): open a new RX at the new frequency")

    def start(self):
        cmd = self.uhd.types.StreamCMD(self.uhd.types.StreamMode.start_cont); cmd.stream_now = True
        self.st.issue_stream_cmd(cmd); self.running = True

    def read(self, n, skip_s=SKIP_S, lag_log=False):
        """Receive n samples after dropping skip_s seconds. Returns complex64 array and the wall-clock start time.
        lag_log=True stores (wall time after each recv, cumulative sample count) in self.lag_rows (save_lag writes them)."""
        if not self.running:
            self.start()
        out = np.empty(n + self.spp, np.complex64)
        buf = np.zeros((1, self.spp), np.complex64)
        drop = int(skip_s * self.rate); got = 0; t0 = None
        while got < n:
            k = self.st.recv(buf, self.md, 1.0)
            ec = self.md.error_code
            if ec == self.uhd.types.RXMetadataErrorCode.overflow:
                self.overflows += 1
            elif ec == self.uhd.types.RXMetadataErrorCode.timeout:
                self.timeouts += 1
                if self.timeouts > 20:
                    raise RuntimeError("20 RX timeouts: no stream from the V-USRP")
            if k <= 0:
                continue
            self.total += k
            if lag_log:
                self.lag_rows.append((time.time(), self.total))
            if drop > 0:
                d = min(drop, k); drop -= d
                if d == k:
                    continue
                out[got:got + k - d] = buf[0, d:k]; got += k - d
            else:
                out[got:got + k] = buf[0, :k]; got += k
            if t0 is None:
                t0 = time.time() - got / self.rate
        return out[:n], t0

    def save_lag(self, path):
        """Write the lag log as a .npy file with two columns: wall time (s) and cumulative sample count."""
        np.save(path, np.array(self.lag_rows, float)); return path

    def chunks(self, block, seconds, skip_s=SKIP_S):
        """Generator of blocks of `block` samples for `seconds` (one continuous stream)."""
        x, t0 = self.read(block, skip_s); yield x, t0
        for i in range(1, int(seconds * self.rate / block)):
            x, t = self.read(block, 0.0); yield x, t

    def stop(self):
        if self.running:
            self.st.issue_stream_cmd(self.uhd.types.StreamCMD(self.uhd.types.StreamMode.stop_cont)); self.running = False


class TX:
    """Continuous transmitter in a thread: repeats one waveform period (zeros = register only, CHEM noise reference)."""

    def __init__(self, rate, freq, gain=60.0):
        self.rate, self.freq, self.gain = rate, freq, gain
        self.uhd, self.u = _usrp(rate, freq, gain, "tx")
        sa = self.uhd.usrp.StreamArgs("fc32", "sc16"); sa.channels = [0]
        self.st = self.u.get_tx_stream(sa)
        self.chunk = min(10 * self.st.get_max_num_samps(), SEND_MAX_BYTES // 8 // 2)  # fc32 host buffer << 400 kB
        self.sent = 0; self._stop = threading.Event(); self.th = None

    def set_freq(self, *_):
        raise RuleError("no retune while streaming: start a new TX")

    def set_gain(self, g):  # a gain change acts at once and is allowed (L17)
        self.u.set_tx_gain(g, 0); self.gain = g

    def start(self, wave, seconds=None):
        w = np.asarray(wave, np.complex64)
        if np.max(np.abs(w)) > 1.0:
            raise RuleError("waveform peak above full scale")
        reps = int(np.ceil(self.chunk / len(w))) + 1
        tile = np.tile(w, reps)  # a buffer we can slice cyclically
        def run():
            md = self.uhd.types.TXMetadata(); md.start_of_burst = True; md.end_of_burst = False; md.has_time_spec = False
            pos = 0; L = len(w); t0 = time.time()
            while not self._stop.is_set() and (seconds is None or time.time() - t0 < seconds):
                buf = tile[pos:pos + self.chunk].reshape(1, -1)
                n = self.st.send(buf, md); md.start_of_burst = False
                self.sent += n; pos = (pos + n) % L
        self.th = threading.Thread(target=run, daemon=True); self.th.start()
        return self

    def wait(self):
        while self.th and self.th.is_alive():
            time.sleep(0.2)

    def stop(self):
        self._stop.set()
        if self.th:
            self.th.join(2.0)


def with_zero_fill(bursts, gap_samples):
    """Concatenate bursts with zeros between them: the V-USRP needs one continuous stream (L9)."""
    z = np.zeros(int(gap_samples), np.complex64)
    return np.concatenate([np.concatenate([np.asarray(b, np.complex64), z]) for b in bursts])


# ---------------------------------------------------------------- files
def sigmf_paths(base):
    base = base[:-len(".sigmf-data")] if base.endswith(".sigmf-data") else base[:-len(".sigmf-meta")] if base.endswith(".sigmf-meta") else base
    return base + ".sigmf-data", base + ".sigmf-meta"


def write_sigmf(base, x, rate, freq, description="", t0=None, extra_global=None, annotations=(), captures=None):
    """Write cf32_le SigMF (spec 1.2 core fields). Returns the meta path."""
    dpath, mpath = sigmf_paths(base)
    np.asarray(x, np.complex64).tofile(dpath)
    ts = datetime.datetime.fromtimestamp(t0 or time.time(), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    g = {"core:datatype": "cf32_le", "core:sample_rate": float(rate), "core:version": "1.2.0",
         "core:description": description, "core:hw": "AERPAW V-USRP (emulated B205) over CHEM",
         "core:recorder": "aerpaw_py", "core:author": "AERPAW digital twin X1532",
         "core:extensions": [{"name": "aerpaw", "version": "0.1.0", "optional": True}]}
    g.update(extra_global or {})
    caps = captures or [{"core:sample_start": 0, "core:frequency": float(freq), "core:datetime": ts}]
    json.dump({"global": g, "captures": caps, "annotations": list(annotations)}, open(mpath, "w"), indent=1)
    return mpath


def read_iq(path, rate=None):
    """Read SigMF (cf32_le/ci16_le) or raw complex64. Returns x, rate, freq, meta."""
    if path.endswith(".sigmf-meta") or path.endswith(".sigmf-data") or os.path.exists(path + ".sigmf-meta"):
        dpath, mpath = sigmf_paths(path)
        meta = json.load(open(mpath)); g = meta["global"]
        dt = g["core:datatype"]
        if dt == "cf32_le":
            x = np.fromfile(dpath, np.complex64)
        elif dt == "ci16_le":
            r = np.fromfile(dpath, np.int16).astype(np.float32) / 32768; x = (r[0::2] + 1j * r[1::2]).astype(np.complex64)
        else:
            raise ValueError("unsupported datatype " + dt)
        return x, g["core:sample_rate"], meta["captures"][0].get("core:frequency"), meta
    return np.fromfile(path, np.complex64), rate, None, {}


def sigmf_validate(meta_path):
    """Validate with sigmf-python when it is installed in the private _vendor folder. Returns a short text."""
    sys.path.insert(0, VENDOR)
    try:
        import sigmf
        s = sigmf.fromfile(meta_path)
        s.validate()
        return f"sigmf {sigmf.__version__}: valid, {s.sample_count} samples"
    except ImportError:
        return "sigmf-python not installed (skip)"
    except Exception as e:
        return f"sigmf validation error: {type(e).__name__}: {str(e)[:200]}"


# ---------------------------------------------------------------- CHEM
def chem_client():
    """ChemClient from the image's /usr/local/bin/pychem (needs the experiment environment)."""
    import importlib.machinery, importlib.util
    loader = importlib.machinery.SourceFileLoader("pychem", "/usr/local/bin/pychem")
    spec = importlib.util.spec_from_loader("pychem", loader); pc = importlib.util.module_from_spec(spec)
    sys.modules["pychem"] = pc; loader.exec_module(pc)
    return pc.ChemClient(), pc


CHEM_FIELDS = ("distance", "elevation", "azimuth", "plDb", "snr", "snrMeasuredDb", "dopplerEnabled", "dopplerHz",
               "freqOffsetHz", "cirTapCount", "noiseModel", "pathLoss")


def chem_links(c, sel=None, freq_mhz=None):
    """List of dicts (one per link) from get_individual_channels. sel = 'M1>M3' filters by node name parts."""
    out = []
    for f, ls in (c.get_individual_channels() or {}).items():
        if freq_mhz and abs(float(f) - freq_mhz) > 0.5:
            continue
        for L in ls:
            if sel and not (sel.split(">")[0] in L["src"] and sel.split(">")[1] in L["dest"]):
                continue
            d = {"freqMHz": float(f), "src": L["src"], "dest": L["dest"]}
            d.update({k: L.get(k) for k in CHEM_FIELDS})
            out.append(d)
    return out


def port_open(host, port, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ChemPoller(threading.Thread):
    """Polls CHEM link state every `period` s into rows (time, link fields); writes a CSV on save()."""

    def __init__(self, sel=None, period=0.25, freq_mhz=None):
        super().__init__(daemon=True)
        self.sel, self.period, self.freq_mhz = sel, period, freq_mhz
        self.rows = []; self._stop = threading.Event(); self.err = None
        try:
            self.c, _ = chem_client()
        except Exception as e:
            self.c = None; self.err = str(e)

    def run(self):
        while self.c and not self._stop.is_set():
            try:
                t = time.time()
                for d in chem_links(self.c, self.sel, self.freq_mhz):
                    d["t"] = t; self.rows.append(d)
            except Exception as e:
                self.err = str(e)
            time.sleep(self.period)

    def stop(self):
        self._stop.set()

    def save(self, path):
        cols = ["t", "freqMHz", "src", "dest"] + list(CHEM_FIELDS)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow(r) for r in self.rows]
        return path


def load_chem_csv(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


if __name__ == "__main__":  # rule self-test (no radio): python3 vusrp_io.py
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    tests = {"rate 0.5 MS/s": lambda: check_rate(0.5e6), "rate 30.72 MS/s": lambda: check_rate(30.72e6), "rate 2 MS/s": lambda: check_rate(2e6),
             "TX 3500 MHz": lambda: check_tx_freq(3.5e9), "TX 3400 MHz": lambda: check_tx_freq(3.4e9),
             "TX 914 MHz": lambda: check_tx_freq(914e6), "TX 433.92 MHz": lambda: check_tx_freq(433.92e6),
             "TX 1000 MHz": lambda: check_tx_freq(1000e6)}
    out = {}
    for k, f in tests.items():
        try:
            f(); out[k] = "accepted"
        except RuleError as e:
            out[k] = "refused"
    z = with_zero_fill([np.ones(10)], 5); out["zero_fill_len"] = len(z)
    out["env_sourced"] = env_ok(); out["node"] = node_meta()
    print(json.dumps(out))
