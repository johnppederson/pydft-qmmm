"""Persistent VASP process and atomic file exchange for its structure plugin.

Only the serial plugin rank communicates with the Python driver. Requests are
published after embedding inputs are complete; replies are published after the
force plugin has finished. No sockets or partial vasprun.xml parsing are used.
"""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from .vasp_utils import VaspExecutionError

SESSION_ENV = 'PYDFT_QMMM_VASP_SESSION'
STEPS_ENV = 'PYDFT_QMMM_VASP_STEPS'
TIMEOUT_ENV = 'PYDFT_QMMM_VASP_TIMEOUT'


def publish(path: Path, **data) -> None:
    """Publish a complete NumPy record atomically on the same filesystem."""
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez(handle, **data)
    temporary.replace(path)


class PersistentVasp:
    """Own one VASP process for a bounded sequence of force evaluations."""

    def __init__(self, command, directory, env, steps, timeout):
        self.directory = Path(directory).resolve()
        self.exchange = self.directory / ('.pydft-session-' + uuid.uuid4().hex)
        self.exchange.mkdir()
        self.steps, self.timeout = steps, timeout
        self.frame = 0
        self.closed = False
        self.process = None
        env = dict(env)
        env.update({
            SESSION_ENV: str(self.exchange), STEPS_ENV: str(steps),
            TIMEOUT_ENV: str(timeout),
        })
        try:
            with (
                (self.directory/'vasp.out').open('w') as out,
                (self.directory/'vasp.err').open('w') as err,
            ):
                self.process = subprocess.Popen(
                    command, shell=True, cwd=self.directory, env=env,
                    stdout=out, stderr=err, start_new_session=True,
                )
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _failure(self, message):
        self.close()
        return VaspExecutionError(str(self.directory), message)

    def evaluate(self, positions):
        """Advance the waiting structure callback, or receive the first SCF."""
        if self.closed or self.frame >= self.steps:
            raise self._failure(
                'Persistent VASP session is closed or exhausted.',
            )
        if self.frame:
            if self.process.poll() is not None:
                raise self._failure('VASP exited before the next geometry.')
            publish(
                self.exchange/f'request_{self.frame:06d}.npz',
                step=self.frame, positions=positions,
            )
        result = self.exchange/f'result_{self.frame:06d}.npz'
        deadline = time.monotonic() + self.timeout
        while not result.is_file():
            code = self.process.poll()
            if code is not None:
                # The final rename can race with process exit on fast jobs.
                if result.is_file():
                    break
                raise self._failure(
                    f'Persistent VASP exited with code {code} before '
                    f'frame {self.frame}.',
                )
            if time.monotonic() >= deadline:
                raise self._failure(
                    f'Timed out waiting for VASP frame {self.frame}.',
                )
            time.sleep(.05)
        try:
            with np.load(result, allow_pickle=False) as data:
                if int(data['step']) != self.frame:
                    raise ValueError('Incorrect frame number in VASP reply')
                energy = float(data['energy'])
                forces = np.array(data['forces'], copy=True)
            if (
                forces.shape != positions.shape
                or not np.isfinite(forces).all()
                or not np.isfinite(energy)
            ):
                raise ValueError('Invalid energy or force array in VASP reply')
            result.unlink()
            self.frame += 1
            if self.frame == self.steps:
                code = self.process.wait(timeout=self.timeout)
                if code:
                    raise ValueError(
                        f'VASP exited with code {code} after its final frame',
                    )
                self.close()
            return energy, forces
        except BaseException:
            self.close()
            raise

    def close(self):
        """Reap VASP; early close terminates the MPI process group.

        Use exactly ``persistent_steps`` evaluations for normal VASP shutdown
        and final restart files. Early close does not promise a checkpoint.
        """
        if self.closed:
            return
        self.closed = True
        process = self.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        atexit.unregister(self.close)


# Each VASP process owns its callback counter, independently of driver frames.
_plugin_frame = 0


def structure(constants, additions):
    """Publish energy/forces and receive the next fractional geometry."""
    global _plugin_frame
    exchange = Path(os.environ[SESSION_ENV])
    steps = int(os.environ[STEPS_ENV])
    timeout = float(os.environ[TIMEOUT_ENV])
    frame = _plugin_frame
    publish(
        exchange/f'result_{frame:06d}.npz', step=frame,
        energy=float(constants.total_energy),
        forces=np.asarray(constants.forces),
    )
    _plugin_frame += 1
    if _plugin_frame >= steps:
        # NSW ends this process normally, with final output/checkpoints.
        return
    request = exchange/f'request_{_plugin_frame:06d}.npz'
    deadline = time.monotonic() + timeout
    while not request.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                'Timed out waiting for the next PyDFT-QMMM geometry',
            )
        time.sleep(.05)
    with np.load(request, allow_pickle=False) as data:
        if int(data['step']) != _plugin_frame:
            raise ValueError('Incorrect frame number in geometry request')
        positions = np.array(data['positions'], copy=True)
    request.unlink()
    current = np.asarray(constants.positions)
    if positions.shape != current.shape or not np.isfinite(positions).all():
        raise ValueError('Invalid geometry request')
    # VASP uses fractional coordinates and expects increments, not
    # replacements.
    additions.positions += positions - current
    # The external potential depends on both positions and the II/III
    # partition. Its grid shape can stay fixed even though every MM source
    # has changed.
    from .vasp_plugin import reset_cache
    reset_cache()
