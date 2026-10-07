# SPDX-License-Identifier: GPL-3.0-or-later
"""Replay a recorded OpticFilm 7600i v1 job (``control`` / ``write`` / ``read`` / ``delay`` ops).

Recorded poll counts are replaced by live waits: write acknowledge, bulk completion, motor idle
before a start, FEEDFSH after the job's own move, BUFEMPTY before image reads.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pyopticfilm.exceptions import AsicError, ScanCancelled, ScanError

STATUS = 0x41
MOTORENB = 0x01
FEEDFSH = 0x20
BUFEMPTY = 0x40
CHUNK = 0x40000
PACKET = 512
THROUGHPUT_GRACE_S = 6.0
THROUGHPUT_MIN = 0.9

_ADDR_OP = {"kind": "control", "rt": 0x40, "request": 0x0C, "value": 0x83, "index": 0, "data": [STATUS]}
_ACK_OP = {"kind": "control", "rt": 0xC0, "request": 0x0C, "value": 0x8E, "index": 0x20, "length": 1}


def _writes_motor_start(op: dict) -> bool:
    d = op.get("data") or []
    return op.get("rt") == 0x40 and op.get("value") == 0x83 and len(d) > 1 and any(
        d[i] == 0x0F and d[i + 1] == 1 for i in range(0, len(d) - 1, 2)
    )


def _table(data: str) -> list[int]:
    b = base64.b64decode(data)
    return [b[i] | (b[i + 1] << 8) for i in range(0, len(b) - 1, 2)]


def prepare_profile(profile: dict, *, dummy_lines: str = "recorded", exposure_multiplier: int = 1) -> dict:
    """Set the main scan's CCD dummy lines (``recorded`` / ``fewer`` / ``none``) or exposure ×k.

    LINESEL L' instead of L scales the main-scan motor cruise to ``C*k*(L'+1)/(L+1)``. Exposure ×k
    also writes LPERIOD×k and BUFSEL 0x08 (k=3 is SilverFast's multi-exposure pass).
    """
    if dummy_lines not in {"recorded", "fewer", "none"} or exposure_multiplier not in {1, 2, 3, 4}:
        raise ValueError("dummy_lines must be recorded/fewer/none and exposure_multiplier 1-4")
    recorded = int(profile["scan"]["lineSel"])
    if exposure_multiplier > 1 or dummy_lines == "none":
        line_sel = 0
    else:
        line_sel = max(0, recorded - 1) if dummy_lines == "fewer" else recorded
    if line_sel == recorded and exposure_multiplier == 1:
        return profile
    k = exposure_multiplier
    ops = profile["ops"]
    main = profile["mainFrame"]
    factor = k * (line_sel + 1) / (recorded + 1)
    first_read = next(i for i, o in enumerate(ops) if o["kind"] == "read" and o["frame"] == main)
    start = next((i for i in range(first_read - 1, -1, -1) if _writes_motor_start(ops[i])), -1)
    prev_read = next((i for i in range(start - 1, -1, -1) if ops[i]["kind"] == "read"), -1)
    if not start > prev_read >= 0:
        raise AsicError("profile has no recognisable main-scan start")
    slot: int | None = None
    tables: list[int] = []
    for i in range(start):
        o = ops[i]
        if o["kind"] == "control" and o["rt"] == 0x40 and o["value"] == 0x83 and len(o["data"]) > 1:
            for j in range(0, len(o["data"]), 2):
                if o["data"][j] == 0x5B:
                    v = o["data"][j + 1]
                    slot = (v >> 3) & 7 if v & 0x40 else None
        if o["kind"] == "write" and i > prev_read and slot is not None and slot <= 2:
            tables.append(i)
    if not tables:
        raise AsicError("profile has no main-scan motor table")
    cruise = _table(ops[tables[0]]["data"])[-1]
    nxt = cruise * factor
    if nxt != int(nxt) or not 0 < nxt <= 0xFFFF:
        raise AsicError("dummy line change does not give a whole motor period")
    nxt = int(nxt)
    line_reg = profile["frames"][main]["regs"].get("30")
    if line_reg is None or line_reg & 0x0F != recorded:
        raise AsicError("recorded main-scan LINESEL not found")
    new_ops = list(ops)
    for i in tables:
        t = _table(ops[i]["data"])
        if t[-1] != cruise:
            raise AsicError("main-scan tables disagree")
        payload = b"".join(bytes((v & 0xFF, v >> 8)) for v in (nxt if v == cruise else v for v in t))
        new_ops[i] = {**ops[i], "data": base64.b64encode(payload).decode()}
    lp = int(profile["scan"]["lPeriod"]) * k
    changed = {0x1E: (line_reg & 0xF0) | line_sel}
    if k > 1:
        changed.update({0x38: lp >> 8, 0x39: lp & 0xFF, 0x20: 0x08})
    d = new_ops[start]["data"]
    data: list[int] = []
    for j in range(0, len(d), 2):
        if d[j] == 0x0F:  # written in the start write, just before 0x0F=1
            data += [b for item in changed.items() for b in item]
        data += [d[j], d[j + 1]]
    new_ops[start] = {**new_ops[start], "data": data}
    frames = list(profile["frames"])
    frames[main] = {**frames[main], "regs": {**frames[main]["regs"], **{str(a): v for a, v in changed.items()}}}
    scan = {**profile["scan"], "lineSel": line_sel, "lPeriod": lp,
            "lineSeconds": round(profile["scan"]["lineSeconds"] * factor, 6),
            "seconds": round(profile["scan"]["seconds"] * factor, 1),
            "bytesPerSecond": round(profile["scan"]["bytesPerSecond"] / factor)}
    return {**profile, "ops": new_ops, "frames": frames, "scan": scan,
            "motorCruise": {"recorded": cruise, "used": nxt}}


def collapse_recorded_waits(ops: list[dict]) -> list[dict]:
    """Collapse 5+ identical recorded status polls with MOTORENB before the first motor start.

    The iSRD job starts while SilverFast's carriage is still returning; the poll becomes one read
    that waits only while the motor really runs.
    """
    first = next((i for i, o in enumerate(ops) if o["kind"] == "control" and _writes_motor_start(o)), -1)

    def triplet(i: int) -> int | None:
        if i + 2 >= len(ops):
            return None
        a, b, c = ops[i : i + 3]
        if (a["kind"] == b["kind"] == c["kind"] == "control" and a.get("data") == [STATUS] and a["value"] == 0x83
                and b["value"] == 0x8E and b["index"] == 0x20 and c["value"] == 0x84
                and c.get("register") == STATUS and c["expected"][0] & MOTORENB):
            return c["expected"][0]
        return None

    out: list[dict] = []
    i = 0
    while i < len(ops):
        e = triplet(i) if i < first else None
        if e is None:
            out.append(ops[i])
            i += 1
            continue
        n = 1
        while i + 3 * n < first and triplet(i + 3 * n) == e:
            n += 1
        if n < 5:
            out += ops[i : i + 3 * n]
        else:
            out += [ops[i], ops[i + 1], {**ops[i + 2], "expected": [e & ~MOTORENB], "waitMotorIdle": n}]
        i += 3 * n
    return ops if len(out) == len(ops) else out


def validate_profile(profile: dict) -> None:
    """Refuse a profile whose frames are incomplete or whose image reads outlast the scan."""
    planned: dict[int, int] = {}
    frames = profile["frames"]
    main = profile["mainFrame"]
    for op in profile["ops"]:
        if op["kind"] == "read":
            planned[op["frame"]] = planned.get(op["frame"], 0) + op["length"]
            if planned[op["frame"]] > frames[op["frame"]]["bytes"]:
                raise AsicError("invalid frame read budget")
        elif op["kind"] == "control" and 0 < planned.get(main, 0) < frames[main]["bytes"]:
            d = op.get("data") or []
            done = op["rt"] == 0xC0 and op["value"] == 0x8E and op["index"] == 0x18
            shutdown = op["rt"] == 0x40 and op["value"] == 0x83 and len(d) > 1 and any(
                (d[i] == 1 and not d[i + 1] & 1) or (d[i] == 3 and not d[i + 1] & 0x10)
                for i in range(0, len(d) - 1, 2)
            )
            if done or shutdown:
                raise AsicError("main frame incomplete before transfer completion or shutdown")
    if any(planned.get(i) != f["bytes"] for i, f in enumerate(frames)):
        raise AsicError("incomplete frame in profile")


@dataclass
class ReplayHooks:
    check: Callable[[], None] = lambda: None
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    progress: Callable[[int, int, int], None] | None = None
    #: called with (frame, bytes) when a calibration frame (white line, dark, white) is complete
    frame_done: Callable[[int, bytes], None] | None = None
    timeout_s: float = 60.0
    read_chunk: int = CHUNK


@dataclass
class ReplayResult:
    main: bytearray
    frames: dict[int, bytes] = field(default_factory=dict)
    positioning_stop_ms: float | None = None


def run_profile(profile: dict, usb, hooks: ReplayHooks | None = None) -> ReplayResult:
    """Replay ``profile`` on ``usb`` (:class:`~pyopticfilm.asic.gl843_v1.Gl843V1Usb`)."""
    h = hooks or ReplayHooks()
    validate_profile(profile)
    transport = usb.transport
    frames = profile["frames"]
    main = profile["mainFrame"]
    lamp = profile.get("lamp") or {}
    calib_frames = {lamp[k]["frame"] for k in ("line", "shading", "dark") if k in lamp}
    need = profile["scan"]["bytesPerSecond"]
    regs: dict[int, int] = {}
    st = {"address": 0, "status": None, "moved": None}
    buffers = {i: bytearray(frames[i]["bytes"]) for i in calib_frames | {main}}
    got = dict.fromkeys(range(len(frames)), 0)
    result = ReplayResult(main=buffers[main])
    pool = ThreadPoolExecutor(max_workers=1)

    def raw(op: dict) -> bytes:
        if op["rt"] == 0x40:
            transport.control_msg(0x40, op["request"], op["value"], op["index"], bytes(op["data"]))
            if op["value"] == 0x83:
                d = op["data"]
                if len(d) == 1:
                    st["address"] = d[0]
                for i in range(0, len(d) - 1, 2):
                    regs[d[i]] = usb.shadow[d[i]] = d[i + 1]
            return b""
        r = bytes(transport.control_msg(0xC0, op["request"], op["value"], op["index"], op["length"]))
        if len(r) != op["length"]:
            raise ScanError("short control read")
        if op["value"] == 0x84:  # the GL843 auto-increments the address after a read
            if st["address"] == STATUS:
                st["status"] = r[0]
            st["address"] = (st["address"] + 1) & 0xFF
        return r

    def wait(r: bytes, again: Callable[[], bytes], ready: Callable[[int], bool], why: str, pause: float) -> None:
        deadline = h.now() + h.timeout_s
        while not ready(r[0]):
            h.check()
            if h.now() > deadline:
                raise ScanError(f"timeout {why} (status 0x{r[0]:02x})")
            h.sleep(pause)
            r = again()

    def ack() -> bytes:
        return raw(_ACK_OP)

    def poll(op: dict, r: bytes, mask: int, target: int, why: str) -> None:
        def status() -> bytes:
            raw(_ADDR_OP)
            wait(ack(), ack, lambda s: bool(s & 1), "waiting for write acknowledge", 0.005)
            return raw(op)

        wait(r, status, lambda s: s & mask == target, why, 0.015)

    def control(op: dict) -> None:
        if op["rt"] == 0x40:
            start = _writes_motor_start(op)
            if start and st["status"] is not None and st["status"] & MOTORENB:
                poll({**_ACK_OP, "index": 0, "value": 0x84}, bytes([st["status"]]), MOTORENB, 0, "motor before start")
            raw(op)
            if start:
                st["status"], st["moved"] = None, h.now()
            return
        r = raw(op)
        if op["value"] == 0x8E and op["index"] == 0x20:
            wait(r, lambda: raw(op), lambda s: bool(s & 1), "waiting for write acknowledge", 0.005)
        elif op["value"] == 0x8E and op["index"] == 0x18:
            wait(r, lambda: raw(op), lambda s: not s & 0x0C, "waiting for bulk completion", 0.02)
        elif op.get("waitMotorIdle"):
            poll(op, r, MOTORENB, 0, "waiting for the previous move")
        elif op["value"] == 0x84 and op.get("register") == STATUS:
            expected = op["expected"][0]
            scanning = regs.get(1, 0) & 1
            if not scanning and expected & (FEEDFSH | MOTORENB) == FEEDFSH:
                # FEEDFSH latches the end of a move; before this job's own move it is the vendor's.
                if st["moved"] is None:
                    poll(op, r, MOTORENB, 0, "waiting for the previous move")
                else:
                    poll(op, r, FEEDFSH | MOTORENB, FEEDFSH, "waiting for the positioning move")
            elif scanning and not expected & BUFEMPTY:
                poll(op, r, BUFEMPTY, 0, "waiting for scan data")

    def read(op: dict) -> None:
        i = op["frame"]
        size = frames[i]["bytes"]
        left = asked = op["length"]

        def request():
            # whole 512-byte packets, remainder on its own (a read ending mid-packet overflows)
            nonlocal asked
            n = min(h.read_chunk, asked)
            n = n - n % PACKET if n >= PACKET else n
            asked -= n
            return pool.submit(lambda: (bytes(transport.bulk_read(n)), n))

        pending = request()  # keep one transfer in flight so the scanner's buffer never fills
        t0 = None
        since = 0
        while left:
            h.check()
            part, n = pending.result()
            if not part or len(part) > n or got[i] + len(part) > size:
                raise ScanError(f"frame {i}: bulk read returned {len(part)} of {n} bytes")
            left -= len(part)
            asked += n - len(part)  # a short read is asked for again
            if asked:
                pending = request()
            if i in buffers:
                buffers[i][got[i] : got[i] + len(part)] = part
            got[i] += len(part)
            if t0 is None:
                t0 = h.now()
            elif i == main:
                since += len(part)
                secs = h.now() - t0
                if secs > THROUGHPUT_GRACE_S and since / secs < need * THROUGHPUT_MIN:
                    raise ScanError(f"scan data at {since / secs / 1e6:.2f} MB/s, the scan needs "
                                    f"{need / 1e6:.2f} MB/s")
            if h.progress is not None:
                h.progress(i, got[i], size)
        if i != main and i in buffers and got[i] == size:
            result.frames[i] = bytes(buffers[i])
            if h.frame_done is not None:
                h.frame_done(i, result.frames[i])

    try:
        for op in collapse_recorded_waits(profile["ops"]):
            h.check()
            if op["kind"] == "delay":
                if not op.get("timedStop"):
                    h.sleep(op["ms"] / 1000)
                    continue
                # The vendor stops the first positioning move on the position-sensor event,
                # 2.56-2.57 s after the start; stop at the recorded moment.
                if st["moved"] is None:
                    raise AsicError("timed stop without a preceding motor start")
                target = st["moved"] + op["ms"] / 1000
                while h.now() < target:
                    h.check()
                    h.sleep(max(0.001, min(0.02, target - h.now())))
                result.positioning_stop_ms = (h.now() - st["moved"]) * 1000
            elif op["kind"] == "control":
                h.check()
                control(op)
            elif op["kind"] == "write":
                payload = base64.b64decode(op["data"])
                if transport.bulk_write(payload) != len(payload):
                    raise ScanError("short bulk upload")
            else:
                read(op)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if any(got[i] != f["bytes"] for i, f in enumerate(frames)):
        raise ScanError("incomplete frame")
    return result


def check_cancel(cancel) -> Callable[[], None]:
    def check() -> None:
        if cancel is not None and cancel.is_set():
            raise ScanCancelled("cancelled")

    return check
