"""Utilities for logging energies and positions and related types.
"""

from __future__ import annotations

__all__ = [
    "make_csv_handler",
    "make_log_handler",
    "make_dcd_handler",
    "Loggable",
]

import array
import logging
import os
import pathlib
import re
import struct
from typing import cast
from typing import TYPE_CHECKING

import numpy as np

from . import lattice
from .misc import check_array

if TYPE_CHECKING:
    from typing import Any
    from numpy.typing import NDArray


class FrameRecord(logging.LogRecord):
    """A log record with a frame data.

    Attributes:
        frame: The current frame of a simulation.
    """
    frame: int


class EnergyRecord(FrameRecord):
    """A log record with an energy data.

    Attributes:
        energy: The current energy of a simulation.
    """
    energy: dict[str, Any]


class PositionRecord(FrameRecord):
    """A log record with an position data.

    Attributes:
        position: The current positions of a simulation.
        box: The current box vectors of a simulation.
    """
    positions: NDArray[np.float64]
    box: NDArray[np.float64]


def _flatten_dict(
        dictionary: dict[str, Any],
) -> dict[str, float]:
    """Create a 'flat' version of a nested dictionary.

    Args:
        dictionary: The nested dictionary to flatten.

    Returns:
        A flattened version of the nested dictionary.
    """
    flat = {}
    for key, val in dictionary.items():
        flat.update(
            _flatten_dict(val) if isinstance(val, dict) else {key: val},
        )
    return flat


class PyDFTQMMMIntervalFilter(logging.Filter):
    """Logging filter that grabs logs at a given interval.

    Args:
        interval: The interval at which to write logs in terms of
            simulation steps.
    """

    def __init__(self, interval: int) -> None:
        self.interval = interval

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether or not to log the record.

        Args:
            record: The record to log if the simulation frame is
                divisible by the write interval.

        Returns:
            Whether or not to log the record.
        """
        if not hasattr(record, "frame"):
            return False
        return not record.frame % self.interval


class PyDFTQMMMEnergyFilter(PyDFTQMMMIntervalFilter):
    """Logging filter that grabs logs with an energy attribute.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether or not to log the record.

        Args:
            record: The record to log if the simulation frame is
                divisible by the write interval and if the log
                contains the simulation energy.

        Returns:
            Whether or not to log the record.
        """
        if not super().filter(record):
            return False
        if not hasattr(record, "energy"):
            return False
        return True


class PyDFTQMMMPositionFilter(PyDFTQMMMIntervalFilter):
    """Logging filter that grab logs with position data.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Determine whether or not to log the record.

        Args:
            record: The record to log if the simulation frame is
                divisible by the write interval and if the log
                contains the simulation position data.

        Returns:
            Whether or not to log the record.
        """
        if not super().filter(record):
            return False
        if not hasattr(record, "positions"):
            return False
        if not hasattr(record, "box"):
            return False
        if check_array(record.positions):
            raise TypeError
        return True


class PyDFTQMMMCSVFormatter(logging.Formatter):
    """Logging formatter that formats energy outputs to CSV files.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record data for output to a CSV file.

        Args:
            record: The log record to format.

        Returns:
            The message to write to a CSV file.
        """
        record = cast(EnergyRecord, record)
        flat_components = _flatten_dict(record.energy)
        round_ = "3"
        if ((formatting := re.search(r"\.(\d+)f", self._style._fmt))
                is not None):
            round_ = formatting.group(1)
        message = ",".join(
            map(lambda x: f"{x:.{round_}f}", flat_components.values()),
        ) + "\n"
        if not record.frame:
            message = ",".join(flat_components.keys()) + "\n" + message
        return message


class PyDFTQMMMLogFormatter(logging.Formatter):
    """Logging formatter that formats energy outputs to log files.
    """

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
        message = ""
        for i, (key, val) in enumerate(energy.items()):
            if isinstance(val, dict):
                message += self._unwrap_energy(
                    val, spaces + 1,
                    cont + [spaces-1] if i != len(energy)-1 else cont,
                )
            else:
                round_ = "3"
                if ((formatting := re.search(r"\.(\d+)f", self._style._fmt))
                        is not None):
                    round_ = formatting.group(1)
                value = f"{val:.{round_}f} kJ/mol\n"
                if spaces:
                    key = "".join(
                        "| " if i in cont else "  "
                        for i in range(spaces - 1)
                    ) + "|_" + key
                message += f"{key}:{value: >{72-len(key)}}"
        return message

    def format(
            self,
            record: logging.LogRecord,
            spaces: int = 0,
            cont: list[int] = [],
    ) -> str:
        """Format the log record data for output to a log file.

        Args:
            record: The log record to format.

        Returns:
            The message to write to a log file.
        """
        record = cast(EnergyRecord, record)
        message = ""
        if not record.frame:
            message += f"{' PyDFT-QMMM Logger ':=^72}\n"
        message += f"{' Frame ' + f'{record.frame:0>6}' + ' ':-^72}\n"
        message += self._unwrap_energy(record.energy)
        return message


class DCDHandler(logging.Handler):
    r"""Handler for writing position data to a DCD file.

    Args:
        filename: The path of the DCD file.
        interval: The interval between subsequent writes in terms
            of simulation steps.
        timestep: The timestep (:math:`\mathrm{fs}`) of the simulation.
        mode: The mode in which to access the DCD file.
    """

    def __init__(
            self,
            filename: str | bytes | os.PathLike,
            interval: int,
            timestep: int | float,
            mode: str = "w+b",
    ) -> None:
        self.interval = interval
        self.timestep = timestep
        super().__init__()
        self.stream = open(filename, mode)

    def _build_header(self, system_size: int) -> bytes:
        """Generate the header to the DCD file.

        Args:
            system_size: The number of particles in the system.

        Returns:
            The header of the DCD file in binary format.
        """
        header = struct.pack(
            "<i4c9if", 84, b"C", b"O", b"R", b"D",
            0, 0, self.interval, 0, 0, 0, 0, 0, 0, self.timestep,
        )
        header += struct.pack(
            "<13i", 1, 0, 0, 0, 0, 0, 0, 0, 0, 24,
            84, 164, 2,
        )
        header += struct.pack("<80s", b"Created by PyDFT-QMMM")
        header += struct.pack("<80s", b"Created now")
        header += struct.pack("<4i", 164, 4, system_size, 4)
        return header

    def emit(self, record: logging.LogRecord) -> None:
        """Write a record to a DCD file.

        Args:
            record: The record to write to a DCD file.
        """
        record = cast(PositionRecord, record)
        try:
            self.acquire()
            if self.stream.closed:
                self.stream = open(self.stream.name, self.stream.mode)
            positions, box, frame = record.positions, record.box, record.frame
            a, b, c, A, B, G = lattice.compute_lattice_constants(box)
            system_size = len(positions)
            if frame == 0:
                header = self._build_header(system_size)
                self.stream.write(header)
            self.stream.seek(8, os.SEEK_SET)
            self.stream.write(struct.pack("<i", frame//self.interval))
            self.stream.seek(20, os.SEEK_SET)
            self.stream.write(struct.pack("<i", frame))
            self.stream.seek(0, os.SEEK_END)
            self.stream.write(struct.pack("<i6di", 48, a, G, b, B, A, c, 48))
            size = struct.pack("<i", 4*system_size)
            for i in range(3):
                self.stream.write(size)
                coordinate = array.array("f", (r[i] for r in positions))
                coordinate.tofile(self.stream)
                self.stream.write(size)
            self.stream.flush()
        except Exception:
            self.handleError(record)
        finally:
            self.release()

    def close(self) -> None:
        """Close the DCD file stream."""
        try:
            self.acquire()
            try:
                self.stream.close()
            finally:
                super().close()
        finally:
            self.release()


def make_file_handler(
        output_directory: str,
        suffix: str,
        formatter: logging.Formatter,
        filter_: logging.Filter,
) -> logging.Handler:
    """Create a handler for logging to files.

    Args:
        output_directory: A directory where records are written.
        suffix: A file extension.
        formatter: A formatter to apply to records.
        filter_: A filter to apply to records.

    Returns:
        The file handler applying the specified filters and formatters.
    """
    outfile = pathlib.Path(output_directory) / ("pydft_qmmm" + suffix)
    handler = logging.FileHandler(outfile)
    handler.addFilter(filter_)
    handler.setFormatter(formatter)
    return handler


def make_csv_handler(
        output_directory: str,
        decimal_places: int = 3,
        interval: int = 1,
) -> logging.Handler:
    """Create a handler for logging to CSV files.

    Args:
        output_directory: A directory where records are written.
        decimal_places: The number of decimal places to include when
            logging to a CSV file.
        interval: The interval at which to write logs in terms of
            simulation steps.

    Returns:
        A CSV file handler.
    """
    handler = make_file_handler(
        output_directory,
        ".csv",
        PyDFTQMMMCSVFormatter(f"%(message).{decimal_places}f"),
        PyDFTQMMMEnergyFilter(interval),
    )
    return handler


def make_log_handler(
        output_directory: str,
        decimal_places: int = 3,
        interval: int = 1,
) -> logging.Handler:
    """Create a handler for logging to log files.

    Args:
        output_directory: A directory where records are written.
        decimal_places: The number of decimal places to include when
            logging to a log file.
        interval: The interval at which to write logs in terms of
            simulation steps.

    Returns:
        A log file handler.
    """
    handler = make_file_handler(
        output_directory,
        ".log",
        PyDFTQMMMLogFormatter(f"%(message).{decimal_places}f"),
        PyDFTQMMMEnergyFilter(interval),
    )
    return handler


def make_dcd_handler(
        output_directory: str,
        interval: int = 1,
        timestep: int | float = 1,
) -> logging.Handler:
    r"""Create a handler for logging to DCD files.

    Args:
        output_directory: A directory where records are written.
        interval: The interval at which to write logs in terms of
            simulation steps.
        timestep: The timestep (:math:`\mathrm{fs}`) of the simulation.

    Returns:
        A DCD file handler.
    """
    outfile = pathlib.Path(output_directory) / "pydft_qmmm.dcd"
    handler = DCDHandler(outfile, interval, timestep)
    handler.addFilter(PyDFTQMMMPositionFilter(interval))
    return handler


class Loggable:
    r"""A mix-in for generating logging handlers.

    Args:
        output_directory: A directory where logs will be written.
        log_write: Whether or not to write energies to a log file.
        log_write_interval: The interval at which to write logs to a
            log file in terms of simulation steps.
        log_decimal_places: The number of decimal places to include
            when logging to a log file.
        csv_write: Whether or not to write energies to a CSV file.
        csv_write_interval: The interval at which to write logs to a
            CSV file in terms of simulation steps.
        csv_decimal_places: The number of decimal places to include
            when logging to a CSV file.
        dcd_write: Whether or not to write positions to a DCD file.
        dcd_write_interval: The interval at which to write logs to a
            DCD file in terms of simulation steps.
        dcd_timestep: The timestep (:math:`\mathrm{fs}`) of the
            simulation.
    """

    def __init__(
            self,
            output_directory: str = ".",
            log_write: bool = True,
            log_write_interval: int = 1,
            log_decimal_places: int = 3,
            csv_write: bool = True,
            csv_write_interval: int = 1,
            csv_decimal_places: int = 3,
            dcd_write: bool = True,
            dcd_write_interval: int = 50,
            dcd_timestep: int | float = 1,
    ) -> None:
        if not pathlib.Path(output_directory).exists():
            os.makedirs(pathlib.Path(output_directory))
        handlers = []
        if log_write:
            log_handler = make_log_handler(
                output_directory,
                log_decimal_places,
                log_write_interval,
            )
            handlers.append(log_handler)
        if csv_write:
            csv_handler = make_csv_handler(
                output_directory,
                csv_decimal_places,
                csv_write_interval,
            )
            handlers.append(csv_handler)
        if dcd_write:
            dcd_handler = make_dcd_handler(
                output_directory,
                dcd_write_interval,
                dcd_timestep,
            )
            handlers.append(dcd_handler)
        self.handlers = handlers
