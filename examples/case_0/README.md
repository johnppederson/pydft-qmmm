Example Case 0
==============

Summary
-------
This case serves as a basic example of the Python API and CLI for the
package.  The system under study is a QM water molecule in a box of MM
water modeled by the SPC/E forcefield.  The QM/MM/Cutoff algorithm is
applied with an electrostatic embedding cutoff of 14.0 Angstroms.
Initial velocities are set according to the Maxwell-Boltzmann
distribution at 300 K using a seed.  Rigid water is enforced through the
SETTLE algorithm, which is implemented as an `IntegratorPlugin`.  The
simulation is run for 10 steps at the 1 fs step-size using a
leap-frog Verlet algorithm.  All files needed to run this case are in
this case directory.

How to Run
----------
The script using the Python API of PyDFT-QMMM can be run with the
following command:

```bash
python case_0_api.py
```

Alternatively, the input file can be read-in and run from the command
line with the following command.

```bash
pydft-qmmm case_0_cli.ini
```

What to Expect
--------------
The logging outputs of the Python API approach will be printed out to
the `./output_api/` subdirectory, and the logging outputs of the CLI
approach will be printed out to the `./output_cli/` subdirectory.
Between the two approaches, the energies should be the same within
numerical error.
