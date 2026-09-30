# SPARC electrostatic QM/MM/PME

Run `python case_3_sparc_electrostatic.py --prepare-only` to check the setup,
or omit `--prepare-only` to calculate energy and forces on allocated
resources. See [the case overview](../README.md) for requirements and launchers.
A SPARC binary with QM/MM embedding support and `helpme-py` are required.

The first water is QM. A molecule-centroid cutoff of 2.5 Å leaves nonempty
regions II and III in this geometry. Periodic Gaussians of width 0.3 Å
supply region II; PME uses only region III charges with no field exclusions.
Physical force-field charges remain unchanged. SPARC applies both electronic
and nuclear embedding. `PMEExcluded` corrects the OpenMM energy without
adding forces for engine coupling.

OpenMM uses PME with a 5 Å cutoff, grid 40³, and α = 5 nm⁻¹.
The QM/MM PME field uses grid 40³ and α = 0.5 Å⁻¹, the same physical α.
SPARC uses PBE, target grid spacing 0.12 Bohr (189³ here), SCF tolerance
1e-8, and density mixing without a preconditioner (parameter 0.3).

Energy and QM forces print to stdout. `--workdir` defaults to
`sparc_workdir`; it contains SPARC outputs and `QMMM_VEXT.bin` and
`QMMM_PHI.bin`. Use shared storage when running across nodes.

## Grid convergence

The separate embedded water gradient benchmark used a 15.6358 Å cell,
247³ grid (target 0.12 Bohr), SCF tolerance 1e-8, and central displacements
of 0.0025 Å. Total gradient RMSE was 0.781 kJ/(mol Å); the SPARC component
RMSE was 0.639 kJ/(mol Å). These are benchmark results, not measured errors
for this four-water example. Isolated-water errors did not decrease
monotonically from 0.12 to 0.10 Bohr. Check force convergence for your
system; accurate interaction energies alone do not establish accurate forces.
