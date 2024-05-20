#! /usr/bin/env python3
"""A module defining the :class:`Logger` class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydft_qmmm.common import align_dict
from pydft_qmmm.common import FileManager

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.system import System
    from .simulation import Simulation


class NullLogger:
    """A default logger class which does not perform any logging.
    """

    def __enter__(self) -> NullLogger:
        """Begin managing the logging context.

        :return: The NullLogger class for context management.
        """
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        """Exit the managed context.

        :param type_: The type of exception raised by the context.
        :param value: The value of the exception raised by the context.
        :param traceback: The traceback from the exception.
        """
        pass

    def record(self, simulation: Simulation) -> None:
        """Default record call, which does nothing.

        :param simulation: |simulation| to record/log.
        """
        pass


@dataclass
class Logger:
    """A logger for writing :class:`Simulation` and :class:`System`
    data.

    :param output_dir: The directory where outputs are written.
    :param system: |system| which will be reported.
    :param write_to_log: Whether or not to write energies to a tree-like
        log file.
    :param decimal_places: Number of decimal places to write energies in
        the log file before truncation.
    :param log_write_interval: The interval between successive log
        writes, in simulation steps.
    :param write_to_csv: Whether or not to write energies to a CSV file.
    :param csv_write_interval: The interval between successive CSV
        writes, in simulation steps.
    :param write_to_dcd: Whether or not to write atom positions to a
        DCD file.
    :param dcd_write_interval: The interval between successive DCD
        writes, in simulation steps.
    :param write_to_pdb: Whether or not to write atom positions to a
        PDB file at the end of a simulation.
    """
    output_directory: str
    system: System
    write_to_log: bool = True
    decimal_places: int = 6
    log_write_interval: int = 1
    write_to_csv: bool = True
    csv_write_interval: int = 1
    write_to_dcd: bool = True
    dcd_write_interval: int = 50
    write_to_pdb: bool = True

    def __enter__(self) -> Logger:
        """Create output files which will house output data from the
        :class:`Simulation`.

        :return: The Logger class with all desired logs initialized.
        """
        self.file_manager = FileManager(self.output_directory)
        if self.write_to_log:
            self.log = "output.log"
            self.file_manager.start_log(self.log)
        if self.write_to_csv:
            self.csv = "output.csv"
            self.file_manager.start_csv(self.csv, "")
        if self.write_to_dcd:
            self.dcd = "output.dcd"
            self.file_manager.start_dcd(
                self.dcd, self.dcd_write_interval, len(self.system), 1,
            )
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        """Perform any termination steps required at the end of the
        :class:`Simulation`.

        :param type_: The type of exception raised by the context.
        :param value: The value of the exception raised by the context.
        :param traceback: The traceback from the exception.
        """
        if self.write_to_log:
            self.file_manager.end_log(self.log)
        if self.write_to_pdb:
            self.file_manager.write_to_pdb(
                "output.pdb",
                self.system.positions,
                self.system.box,
                self.system.molecules,
                self.system.molecule_names,
                self.system.elements,
                self.system.names,
            )

    def record(self, simulation: Simulation) -> None:
        """Log the current state of the :class:`Simulation` to any
        relevant output files.

        :param simulation: |simulation| to log/record.
        """
        if self.write_to_log:
            self.file_manager.write_to_log(
                self.log,
                self._unwrap_energy(simulation.energy),
                simulation.frame,
            )
        if self.write_to_csv:
            flat_dict = align_dict(simulation.energy)
            if simulation.frame > 0:
                self.file_manager.write_to_csv(
                    self.csv,
                    ",".join(
                        f"{val}" for val
                        in flat_dict.values()
                    ),
                )
            else:
                self.file_manager.write_to_csv(
                    self.csv,
                    ",".join(
                        f"{val}" for val
                        in flat_dict.values()
                    ),
                    header=",".join(
                        f"{key}" for key
                        in flat_dict.keys()
                    ),
                )
        if self.write_to_dcd:
            self.file_manager.write_to_dcd(
                self.dcd,
                self.dcd_write_interval,
                len(self.system),
                simulation.system.positions,
                simulation.system.box,
                simulation.frame,
            )

    def _unwrap_energy(
            self,
            energy: dict[str, Any],
            spaces: int = 0,
            cont: list[int] = [],
    ) -> str:
        """Generate the Log string given the :class:`Simulation` energy
        dictionary.

        :param energy: The energy calculated by the :class:`Simulation`
            object.
        :param spaces: The number of spaces to indent the line.
        :parma cont: A list to keep track of sub-component continuation.
        :return: The string to write to the log file.
        """
        string = ""
        for i, (key, val) in enumerate(energy.items()):
            if isinstance(val, dict):
                string += self._unwrap_energy(
                    val, spaces + 1, (
                        cont+[spaces-1] if i != len(energy)-1 else cont
                    ),
                )
            else:
                value = f"{val:.{self.decimal_places}f} kJ/mol\n"
                if spaces:
                    key = "".join(
                        "| " if i in cont else "  "
                        for i in range(spaces - 1)
                    )+"|_"+key
                string += f"{key}:{value: >{72-len(key)}}"
        return string
