Example Case 0
==============

Summary
-------
This case serves as an example of using different MM forcefield and
topology file formats.  The system under study is a QM water molecule in
a box of MM water modeled by the SPC/Fw forcefield.  The QM/MM/Cutoff
algorithm is applied with an electrostatic embedding cutoff of 14.0
Angstroms.  Initial velocities are set according to the Maxwell-Boltzmann
distribution at 300 K using a seed.  The simulation is run for 10 steps
at the 1 fs step-size using a leap-frog Verlet algorithm.  All files
needed to run this case are in this case directory.  Running this case
requires installation of the [ParmEd](https://github.com/ParmEd/ParmEd)
package.

How to Run
----------
The script using FF XML files can be run with the following command:

```bash
python case_2_xml.py
```

The script using the GROMACS top file can be run with the following
command:

```bash
python case_2_top.py
```

The script using the AMBER prmtop/parm7 files can be run with the
following command:

```bash
python case_2_prmtop.py
```

What to Expect
--------------
The logging outputs of the script using FF XML files will be printed
out to the `./output_xml/` subdirectory, the logging outputs of the
script using the GROMACS top file will be printed out to the
`./output_top/` subdirectory, and the logging outputs of the script
using the AMBER prmtop/parm7 files will be printed out to the
`./output_prmtop` subdirectory.  Between the three approaches, the
energies should be the same within numerical error.
