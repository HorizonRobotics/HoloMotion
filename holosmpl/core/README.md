# HoloSMPL Core Code

`core/` contains source-independent logic shared by all datasets and devices:

- `schema/`: canonical and formal HoloSMPL schema definitions.
- `processing/`: coordinate transforms, resampling, pose layout conversion, and quality gates.
- `validation/`: canonical/formal NPZ/H5 validation.
- `writers/`: canonical/formal NPZ/H5 writers.
- `config.py`, `paths.py`, `metadata.py`, `hashing.py`, `io.py`: small shared utilities.

Dataset- or device-specific conversion code should live under `holosmpl/converters/`.
User-facing source notes should live under `holosmpl/supported_datasets/` or
`holosmpl/supported_devices/`.
