# Pico 4 Ultra Enterprise / XRoboToolkit

- Source type: device capture
- Expected input: Pico 4 Ultra Enterprise/XRoboToolkit CSV files
- HoloSMPL source keys: `pico4_ultra_enterprise`, alias `pico`

Pico raw contains global tracker body poses, not fitted SMPL mesh parameters.
The converter assumes SMPL-like 24-joint order, uses the SMPL 24 parent tree to
convert global rotations to local rotations, and writes neutral clip-level beta.
