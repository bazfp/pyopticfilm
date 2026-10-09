# OpticFilm 7600i v1 (GL843)

`07b3:0c3b`, bcdDevice `0x0400`. Capture-derived: each scan replays SilverFast's job sequence for
that resolution, from USB captures of a real 7600i v1 (sequences and findings from
[OpenOptic](https://github.com/bazfp/OpenOptic), where they run on this hardware). The 7600i v2
(bcdDevice `0x0605`, GL845) is unaffected.

| | |
|---|---|
| Colour | 1440, 3600, 7200 dpi full frame, 16-bit linear RGB |
| Infrared | 3600, 7200 dpi (`mode="infrared"`, or `infrared=True` with colour) |
| Crop | `area=`, on the host |
| Not yet | multi-exposure, other resolutions, grey |

Differences from the SANE GL843 path:

- Register reads set the address (`0x83`, one byte) then read `0x84`; the address auto-increments
  after each read. Every write is acknowledged by polling `0x8E`/`0x20` bit 0.
- Boot, AFE, motor tables and shading are the vendor's. The first positioning move is stopped at
  the recorded time (2.56–2.57 s). Homing uses the vendor's motor primitives.
- The image arrives mirrored, with twice as many lines as columns and fractional R/G/B delays;
  7200 dpi has an 8-line column stagger. Infrared is read by the red row.
- Infrared at 7200 dpi was not captured: it is the 7200 dpi colour job with the changes SilverFast
  makes for infrared at 3600 dpi (lamp off, infrared LED, AFE), and shading computed from the job's
  white frame as SilverFast does (`0x13000 × 0x2000 / white`).

Validation: `tests/test_opticfilm_7600i_v1.py` replays every job against strict playback of the
capture (any transfer that differs from SilverFast's fails), and covers boot, homing, image
assembly and an end-to-end `Scanner.open_fake` scan on a simulated GL843.

Limitation: AFE and shading are those of the captured unit. Black-level drift is corrected from
each job's dark frame; other calibration differences are logged.

Data: `device/data/opticfilm_7600i_v1.json.gz`, rebuilt with
`python tools/import_7600i_v1_profiles.py /path/to/OpenOptic`.
