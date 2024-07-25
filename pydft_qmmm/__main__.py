"""
"""
from __future__ import annotations

import re
from argparse import ArgumentParser
from configparser import ConfigParser
from typing import TYPE_CHECKING

import pydft_qmmm.plugins

if TYPE_CHECKING:
    from pydfT_qmmm.plugins.plugin import Plugin


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

    # Check for and adjust interfaces.
    if "Interface" in input_file:
        interface_args = {
            key: value for key, value
            in input_file["Interface"].items()
        }
        pydft_qmmm.set_interfaces(**interface_args)

    # Create System object.
    system_args = {
        key: value for key, value
        in input_file["System"].items()
    }
    if system_args.get("velocities_temperature"):
        temperature = float(system_args.pop("velocities_temperature"))
        velocity_args = [temperature]
        if system_args.get("velocities_seed"):
            seed = int(system_args.pop("velocities_seed"))
            velocity_args.append(seed)
        system = pydft_qmmm.System.load(**system_args)
        velocity_args.insert(0, system.masses)
        system.velocities = pydft_qmmm.generate_velocities(
            *velocity_args,
        )
    else:
        system = pydft_qmmm.System.load(**system_args)
    simulation_args = {"system": system}

    # Create Integrator object.
    if (section := "VerletIntegrator") in input_file:
        integrator_args = {
            key: float(value) for key, value
            in input_file[section].items()
        }
        integrator = getattr(pydft_qmmm, section)(**integrator_args)
    elif (section := "LangevinIntegrator") in input_file:
        integrator_args = {
            key: float(value) for key, value
            in input_file[section].items()
        }
        integrator = getattr(pydft_qmmm, section)(**integrator_args)
    else:
        raise OSError("No Integrator provided in *.ini file.")
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

    # Check for and create Logger object.
    if (section := "Logger") in input_file:
        logger_args = {
            key: _parse_input(value) for key, value
            in input_file[section].items()
        }
        logger = getattr(pydft_qmmm, section)(system=system, **logger_args)
        simulation_args["logger"] = logger

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

    # Create and run the simulation.
    simulation = pydft_qmmm.Simulation(**simulation_args)
    steps = int(input_file["Simulation"]["steps"])
    simulation.run_dynamics(steps)
    return 0
