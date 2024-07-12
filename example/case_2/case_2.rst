==============
Example Case 2
==============

:Author: John Pederson
:Author Email: jpederson6@gatech.edu
:Project: qmmm-pme
:Date Written: September 27, 2023
:Last Date Modified: September 27, 2023

Summary
-------
This case serves as an example of the plugin API for qmmm-pme.  The
system under study is a water molecule in a box of water.  QM/MM is
applied to the water molecule with an electrostatic embedding cutoff of
14.0 Angstroms.  The SETTLE algorithm is applied to all MM atoms, and
this algorithm is applied through a plugin.

How to Run
----------
Assuming that qmmm-pme has been installed correctly, the example can be
run as follows:

``python case_2.py``

What to Expect
--------------
The logging outputs will be printed to an ``/output`` subdirectory in
the current working directory.  Consider viewing trajectory in your
favorite analysis software.
