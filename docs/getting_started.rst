.. _`sec:getting_started`:

===============
Getting Started
===============

Installation
============

PyDFT-QMMM can be installed directly from github using ``pip``:

.. code-block:: bash

    $ python -m pip install git+https://github.com/johnppederson/pydft-qmmm

Alternatively, you can clone the repository and install using ``pip``:

.. code-block:: bash

    $ git clone https://github.com/johnppederson/pydft-qmmm
    $ cd pydft-qmmm
    $ pip install .

Examples
========

Several cases are provided in the example suite.  The CLI and Python API
are demonstrated in :example:`0`.  Enhanced sampling functionality with
Plumed is demonstrated in :example:`1`. Using AMBER or GROMACS forcefield
parameter files is demonstrated in :example:`2`.  

Templates
=========

Templates for writing third-party plugins and interfaces are provided in
the templates folder in the project root directory.
