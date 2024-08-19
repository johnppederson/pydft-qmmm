Example Case 1
==============

Summary
-------
This case serves as a basic example of the Plumed functionality of the
package.  The system under study is a chloride methyl chloride complex
treated at the QM level of theory in a box of MM water modeled by the
TIP3P forcefield.  The QM/MM/Cutoff algorithm is applied with an
electrostatic embedding cutoff of 14.0 Angstroms.  Initial velocities
are set according to the Maxwell-Boltzmann distribution at 300 K using
a seed.  Rigid water is enforced through the SETTLE algorithm, which is
implemented as an `IntegratorPlugin`.  The simulation is run for 1000
steps at the 1 fs step-size using a Langevin algorithm coupled to a
300 K bath and a friction of 0.005 1/fs.  The distance between the first
and second chlorine atoms and the carbon are tracked as collective
variables, and the difference of these distances is subject an umbrella
potential centered at zero.  The coordination of water hydrogens about
the chlorine atoms is also tracked as a collective variable.  All files
needed to run this case are in this case directory.  Running this case
requires installation of the [PLUMED](https://github.com/plumed/plumed2)
package and its Python wrappers.

How to Run
----------
The script can be run with the following command:

```bash
python case_1.py
```

What to Expect
--------------
The logging outputs will be printed out to the `./output/` subdirectory,
and the logging outputs of Plumed will be output to a `COLVAR` file in
the working directory.  The simulation will restrain the chlorine atoms
to be equidistant from the methyl carbon.
