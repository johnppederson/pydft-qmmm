# Mechanical SPARC QM/MM

Run `python case_3_sparc_api.py --prepare-only` to check the setup, or omit
`--prepare-only` to calculate energy and forces on allocated resources.
See [the case overview](../README.md) for installation and launcher commands.

SPARC treats the first water without an external MM field. OpenMM provides
mechanical QM/MM coupling and uses PME with a 5 Å cutoff, 40³ grid, and
α = 5 nm⁻¹. The SPARC grid targets 0.12 Bohr (189³ in this cell), with
SCF tolerance 1e-8 and density mixing without a preconditioner.

Outputs go to `--workdir` (default `sparc_workdir`) and stdout.
The electrostatic example next door uses the same geometry and QM settings.
