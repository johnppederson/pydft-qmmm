"""Centralized logging classes based on context management.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydft_qmmm.common import align_dict
from pydft_qmmm.common import end_log
from pydft_qmmm.common import start_csv
from pydft_qmmm.common import start_dcd
from pydft_qmmm.common import start_log
from pydft_qmmm.common import write_to_csv
from pydft_qmmm.common import write_to_dcd
from pydft_qmmm.common import write_to_log
from pydft_qmmm.common import write_to_pdb

if TYPE_CHECKING:
    from typing import Any
    from pydft_qmmm.system import System
    from .simulation import Simulation


class NullLogger:
    """A default logger class which does not perform logging.
    """

    def __enter__(self) -> NullLogger:
        """Begin managing the logging context.

        Returns:
            A null logger for context management.
        """
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        """Exit the managed context.

        Args:
            type_: The type of exception raised by the context.
            value: The value of the exception raised by the context.
            traceback: The traceback from an exception.
        """
        pass

    def record(self, simulation: Simulation) -> None:
        """Default record call, which does nothing.

        Args:
            simulation: The simulation whose data will be recorded by
                the logger.
        """
        pass


@dataclass
class Logger:
    """Logger for recording system and simulation data.

    Args:
        output_dir: The directory where records are written.
        system: The system whose data will be reported.
        write_to_log: Whether or not to write energies to a tree-like
            log file.
        decimal_places: Number of decimal places to write energies in
            the log file before truncation.
        log_write_interval: The interval between successive log
            writes, in simulation steps.
        write_to_csv: Whether or not to write energies to a CSV file.
        csv_write_interval: The interval between successive CSV
            writes, in simulation steps.
        write_to_dcd: Whether or not to write atom positions to a
            DCD file.
        dcd_write_interval: The interval between successive DCD
            writes, in simulation steps.
        write_to_pdb: Whether or not to write atom positions to a
            PDB file at the end of a simulation.
    """
    output_directory: str
    system: System
    write_to_log: bool = True
    decimal_places: int = 3
    log_write_interval: int = 1
    write_to_csv: bool = True
    csv_write_interval: int = 1
    write_to_dcd: bool = True
    dcd_write_interval: int = 50
    write_to_pdb: bool = True

    def __post_init__(self) -> None:
        """Create output directory if there it does not already exist.
        """
        if not os.path.isdir(self.output_directory):
            os.mkdir(self.output_directory)

    def __enter__(self) -> Logger:
        """Begin managing the logging context.

        This largely entails creating the files which will be logged.

        Returns:
            A logger for context management with access to all necessary
            files in the output directory.
        """
        if self.write_to_log:
            self.log = self.output_directory + "output.log"
            start_log(self.log)
        if self.write_to_csv:
            self.csv = self.output_directory + "output.csv"
            start_csv(self.csv, "")
        if self.write_to_dcd:
            self.dcd = self.output_directory + "output.dcd"
            start_dcd(
                self.dcd, self.dcd_write_interval, len(self.system), 1,
            )
        return self

    def __exit__(self, type_: Any, value: Any, traceback: Any) -> None:
        """Exit the managed context.

        This entails terminating and closing the logging files.

        Args:
            type_: The type of exception raised by the context.
            value: The value of the exception raised by the context.
            traceback: The traceback from an exception.
        """
        if self.write_to_log:
            end_log(self.log)
        if self.write_to_pdb:
            write_to_pdb(
                self.output_directory + "output.pdb",
                self.system,
            )

    def record(self, simulation: Simulation) -> None:
        """Record simulation data into the log files.

        Args:
            simulation: The simulation whose data will be recorded by
                the logger.
        """
        if self.write_to_log:
            write_to_log(
                self.log,
                self._unwrap_energy(simulation.energy),
                simulation._frame,
            )
        if self.write_to_csv:
            flat_dict = align_dict(simulation.energy)
            if simulation._frame > 0:
                write_to_csv(
                    self.csv,
                    ",".join(
                        f"{val}" for val
                        in flat_dict.values()
                    ),
                )
            else:
                write_to_csv(
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
            write_to_dcd(
                self.dcd,
                self.dcd_write_interval,
                self.system,
                simulation._frame,
                simulation._offset,
            )

    def _unwrap_energy(
            self,
            energy: dict[str, Any],
            spaces: int = 0,
            cont: list[int] = [],
    ) -> str:
        """Generate a log file string from an energy dictionary.

        Args:
            energy: The energy component dictionary.
            spaces: The number of spaces to indent the line.
            cont: A list to keep track of sub-component continuation.

        Returns:
            The tree-like string of energies to write to the log file.
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
