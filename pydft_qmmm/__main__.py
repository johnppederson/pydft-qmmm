"""The command line utility.
"""
from __future__ import annotations

import re
from argparse import ArgumentParser
from configparser import ConfigParser
from typing import TYPE_CHECKING

import pydft_qmmm.plugins

if TYPE_CHECKING:
    from typing import TypeAlias
    from pydft_qmmm.calculators import CalculatorPlugin
    from pydft_qmmm.integrators import IntegratorPlugin

    Plugin: TypeAlias = CalculatorPlugin | IntegratorPlugin


def _parse_input(input_value: str) -> int | slice | float | str:
    if re.findall("[a-z, A-Z]", input_value):
        return input_value
    elif re.findall("[:]", input_value):
        elements = input_value.split(":")
        start = None
        end = None
        step = None
        if len(elements) > 2:
            step = int(elements[2])
        if elements[0]:
            start = int(elements[0])
        if elements[1]:
            end = int(elements[1])
        return slice(start, end, step)
    elif re.findall("[.]", input_value):
        return float(input_value)
    else:
        return int(input_value)


def main() -> int:

    # Collect the input file directory.
    parser = ArgumentParser()
    parser.add_argument("input_file")
    args = parser.parse_args()
    # Read input file.
    input_file = ConfigParser()
    input_file.read(args.input_file)

    # Create System object.
    system_args = {
        key: value for key, value
        in input_file["System"].items()
    }
    if system_args.get("pdb_files") is None:
        raise OSError(
            (
                "At least one PDB file must be specified under the [System] "
                "section of the *.ini input file, assigned to the 'pdb_files' "
                "key."
            ),
        )
    pdbs = system_args.get("pdb_files").strip().split("\n")
    system = pydft_qmmm.System.load(*pdbs)
    if system_args.get("velocities_temperature"):
        temperature = float(system_args.pop("velocities_temperature"))
        velocity_args = [temperature]
        if system_args.get("velocities_seed"):
            seed = int(system_args.pop("velocities_seed"))
            velocity_args.append(seed)
        velocity_args.insert(0, system.masses)
        system.velocities = pydft_qmmm.generate_velocities(
            *velocity_args,
        )
    simulation_args = {"system": system}

    # Create Integrator object.
    if (section := "VerletIntegrator") in input_file:
        integrator_args = {
            key: float(value) for key, value
            in input_file[section].items()
        }
    elif (section := "LangevinIntegrator") in input_file:
        integrator_args = {
            key: float(value) for key, value
            in input_file[section].items()
        }
    else:
        raise OSError("No Integrator provided in *.ini file.")
    integrator = getattr(pydft_qmmm, section)(**integrator_args)
    simulation_args["integrator"] = integrator

    # Create Hamiltonian object.
    if (section := "QMHamiltonian") in input_file:
        qm_args = {
            key: _parse_input(value) for key, value
            in input_file[section].items()
        }
        qm_hamiltonian = getattr(pydft_qmmm, section)(**qm_args)
    if (section := "MMHamiltonian") in input_file:
        mm_args = {
            key: _parse_input(value) for key, value
            in input_file[section].items()
        }
        mm_args["forcefield"] = mm_args["forcefield"].strip().split("\n")
        mm_hamiltonian = getattr(pydft_qmmm, section)(**mm_args)
    if (section := "QMMMHamiltonian") in input_file:
        qmmm_args = {
            key: _parse_input(value) for key, value
            in input_file[section].items()
        }
        try:
            qm_region = qmmm_args.pop("region_i")
        except KeyError:
            raise OSError("QMMMHamiltonian field requires region_i defined.")
        qmmm_hamiltonian = getattr(pydft_qmmm, section)(**qmmm_args)
        if qm_region.start:
            hamiltonian = (
                qm_hamiltonian[qm_region.start:qm_region.stop]
                + mm_hamiltonian[0:qm_region.start, qm_region.stop:]
                + qmmm_hamiltonian
            )
        else:
            hamiltonian = (
                qm_hamiltonian[qm_region.start:qm_region.stop]
                + mm_hamiltonian[qm_region.stop:]
                + qmmm_hamiltonian
            )
    elif "MMHamiltonian" in input_file:
        hamiltonian = mm_hamiltonian
    elif "QMHamiltonian" in input_file:
        hamiltonian = qm_hamiltonian
    else:
        raise OSError("No Hamiltonian is provided in *.ini file.")
    simulation_args["hamiltonian"] = hamiltonian

    # Check for and create Plugin objects.
    plugins: list[Plugin] = []
    for section in input_file.sections():
        if section.startswith("Plugins"):
            plugname = section.split(".")[-1]
            plugin_args = {
                key: _parse_input(value) for key, value
                in input_file[section].items()
            }
            plugin = getattr(pydft_qmmm.plugins, plugname)(**plugin_args)
            plugins.append(plugin)
    simulation_args["plugins"] = plugins

    # Check for logging options.
    if (section := "Logging") in input_file:
        logging_args = {
            key: _parse_input(value) for key, value
            in input_file[section].items()
        }
        simulation_args |= logging_args

    # Create and run the simulation.
    simulation = pydft_qmmm.Simulation(**simulation_args)
    steps = int(input_file["Simulation"]["steps"])
    simulation.run_dynamics(steps)
    return 0
