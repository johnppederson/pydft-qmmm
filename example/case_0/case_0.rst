==============
Example Case 0
==============

:Author: John Pederson
:Author Email: jpederson6@gatech.edu
:Project: qmmm-pme
:Date Written: September 27, 2023
:Last Date Modified: September 27, 2023

Summary
-------
This case serves as a basic example of the Python API for qmmm-pme.  The
system under study is a water molecule in a box of BMIM/BF4.  QM/MM is
applied to the water molecule with an electrostatic embedding cutoff of
14.0 Angstroms.

How to Run
----------
Assuming that qmmm-pme has been installed correctly, the example can be
run as follows:

``python case_0.py``

What to Expect
--------------
The logging outputs will be printed to an ``/output`` subdirectory in
the current working directory.
