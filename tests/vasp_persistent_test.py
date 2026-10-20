"""Persistent transport, process lifetime, cache refresh, and input guards."""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pydft_qmmm import Atom
from pydft_qmmm import System
from pydft_qmmm.interfaces.vasp import vasp_persistent as transport
from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp.vasp_interface import VaspPotential
from pydft_qmmm.interfaces.vasp.vasp_utils import VaspExecutionError
from pydft_qmmm.utils import KJMOL_PER_EV
from pydft_qmmm.utils import Subsystem

FAKE_ENGINE = '''
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pydft_qmmm.interfaces.vasp import vasp_plugin
from pydft_qmmm.interfaces.vasp.vasp_persistent import STEPS_ENV
from pydft_qmmm.interfaces.vasp.vasp_persistent import structure

lines = Path('POSCAR').read_text().splitlines()
cell = np.array([[float(x) for x in l.split()] for l in lines[2:5]])
positions = np.array([[float(x) for x in l.split()] for l in lines[8:]])
positions = positions @ np.linalg.inv(cell)
Path('engine.pid').write_text(str(os.getpid()))
for step in range(int(os.environ[STEPS_ENV])):
    assert not vasp_plugin._CACHE
    vasp_plugin._CACHE['stale'] = step
    constants = SimpleNamespace(
        positions=positions, lattice_vectors=cell,
        total_energy=float(np.sum(positions**2)), forces=-2*positions,
    )
    if Path('MM_CHARGES').exists():
        from pydft_qmmm.embedding.grid_potential import read_mm_charges
        from pydft_qmmm.embedding.pme_grid import read_pme_data
        points, charges, stamp, sigma = read_mm_charges('MM_CHARGES')
        assert stamp == step
        _, pme_charges, _, _, _, _, pme_step = read_pme_data('PME_DATA')
        assert pme_step == step
        if Path('SLAB_DATA.json').exists():
            slab = json.loads(Path('SLAB_DATA.json').read_text())
            assert slab['step'] == step
            assert abs(slab['moment']-float(charges @ points[:, 2])) < 1e-12
            Path(f'slab_{step}.json').write_text(json.dumps(slab))
            n_slab = len(slab['charges'])
            Path('SLAB_FORCES').write_text(
                f'{n_slab} {step}\\n' + '0 0 0\\n'*n_slab,
            )
        constants.total_energy += float(
            np.sum(points) + np.sum(charges) + np.sum(pme_charges),
        )
        Path('MM_FORCES').write_text(
            f'{len(points)} {step}\\n' + '1 2 3\\n'*len(points),
        )
        Path('PME_FORCES').write_text(
            f'{len(pme_charges)} {step}\\n' + '4 5 6\\n'*len(pme_charges),
        )
        Path('QM_NET_FORCE').write_text('2\\n0 0 0\\n')
        Path('PLUGIN_FIRED.txt').write_text(str(step))
        with open('PLUGIN_ENERGY', 'a') as fh:
            fh.write('local_potential 0.0\\nforce_and_stress 0.0\\n')
    with open('OUTCAR', 'a') as fh:
        fh.write('aborting loop because EDIFF is reached\\n')
    additions = SimpleNamespace(positions=np.zeros_like(positions))
    structure(constants, additions)
    positions = positions + additions.positions
'''


@pytest.fixture
def potential(tmp_path):
    engine = tmp_path/'fake_engine.py'
    engine.write_text(FAKE_ENGINE)
    pp = tmp_path/'pp'
    for e in ('H', 'O'):
        (pp/e).mkdir(parents=True)
        (pp/e/'POTCAR').write_text('test potential\n')
    # Noncontiguous QM membership and mixed species exercise reordering.
    system = System(
        [
            Atom(element='H', position=np.array(
                [1., 2., 3.]), subsystem=Subsystem.I),
            Atom(
                element='H', position=np.array(
                    [4., 5., 6.],
                ), subsystem=Subsystem.III,
            ),
            Atom(element='O', position=np.array(
                [2., 3., 4.]), subsystem=Subsystem.I),
        ], box=np.eye(3)*10,
    )
    p = VaspPotential(
        system, 0, str(tmp_path/'work'),
        f'{shlex.quote(sys.executable)} {shlex.quote(str(engine))}',
        {'ISYM': 0}, (1, 1, 1), str(pp), {}, persistent=True,
        persistent_steps=3, persistent_timeout=15,
    )
    yield p
    p.close()


def test_reuses_process_and_maps_forces(potential):
    p = potential
    for step in range(3):
        p.system.positions[0, 0] += 0.1
        forces = p.compute_forces()
        expected = np.zeros((3, 3))
        expected[[0, 2]] = -2 * \
            np.asarray(p.system.positions)[[0, 2]]/10*KJMOL_PER_EV
        np.testing.assert_allclose(forces, expected)
        assert p.frame[0] == step+1
        # Cached energy/forces do not request another VASP step.
        assert np.isfinite(p.compute_energy()) and p.frame[0] == step+1
        if step == 0:
            pid = p._session[0].process.pid
        assert p._session[0].process.pid == pid
    assert p._session[0].closed and p._session[0].process.returncode == 0
    assert 'IWAVPR = 11' in (Path(p.directory)/'INCAR').read_text()
    p.system.positions[0, 0] += 0.1
    with pytest.raises(ValueError, match='closed'):
        p.compute_energy()


def test_persistent_slab_refreshes_each_frame(potential):
    """Moving MM sources must update the slab field before each SCF."""
    import json
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    system = System(
        [
            Atom(
                element='H', position=np.array(
                    [1., 2., 3.],
                ), subsystem=Subsystem.I,
            ),
            Atom(
                element='O', position=np.array(
                    [2., 3., 4.],
                ), subsystem=Subsystem.I,
            ),
            Atom(
                element='H', position=np.array(
                    [3., 4., 5.],
                ), charge=1., subsystem=Subsystem.II,
            ),
            Atom(
                element='H', position=np.array(
                    [4., 5., 6.],
                ), charge=-1., subsystem=Subsystem.II,
            ),
        ], box=np.eye(3)*10,
    )
    p = vasp_interface_factory(
        system, pp_path=potential.pp_path, directory=potential.directory,
        command=potential.command, embedding=True, persistent=True,
        persistent_steps=3, persistent_timeout=15,
        incar=dict(ISYM=0, LDIPOL=True, IDIPOL=3, DIPOL=[.5, .5, .5]),
    )
    object.__setattr__(p, 'slab_correction', True)
    p.potentials.append(
        SimpleNamespace(
            pme_alpha=.5, pme_gridnumber=(8, 8, 8),
            pme_spline_order=5,
        ),
    )
    try:
        for step in range(3):
            system.positions[2, 2] += .2
            assert np.isfinite(p.compute_forces()).all()
            slab = json.loads(
                (Path(p.directory)/f'slab_{step}.json').read_text(),
            )
            assert slab['moment'] == pytest.approx(-1 + .2*(step+1))
            assert slab['origin'] == [0., 0., 0.]
            assert slab['qm_self'] is False
            if step == 0:
                pid = p._session[0].process.pid
            assert p._session[0].process.pid == pid
        assert p._session[0].process.returncode == 0
    finally:
        p.close()


@pytest.mark.parametrize('change', ['box', 'membership', 'incar'])
def test_rejects_changes_to_running_hamiltonian(potential, change):
    p = potential
    p.compute_energy()
    if change == 'box':
        p.system.box[0, 0] += 1
    elif change == 'membership':
        p.system.subsystems[1] = Subsystem.I
    else:
        p.incar['ENCUT'] = 600
        p.system.positions[0, 0] += .1
    with pytest.raises(ValueError, match='changed'):
        p.compute_energy()
    assert p._session[0].process.poll() is not None


def test_early_close_reaps_waiting_process(potential):
    potential.compute_energy()
    session = potential._session[0]
    potential.close()
    potential.close()
    assert session.closed and session.process.poll() is not None


def test_persistent_short_cell_coordinates(potential):
    """Requests use local QM fractional positions, while MM remains global."""
    p = potential
    object.__setattr__(p, 'qm_box', np.diag([10., 10., 6.]))
    object.__setattr__(p, 'qm_origin', np.array([0., 0., 1.]))
    object.__setattr__(p, 'embedding_grid_shape', (10, 10, 10))
    p.incar.update(NGXF=10, NGYF=10, NGZF=6)
    for step in range(3):
        p.system.positions[0, 2] += .1
        forces = p.compute_forces()
        local = (
            np.asarray(p.system.positions)[
                [0, 2]
            ] - p.qm_origin
        ) / [10., 10., 6.]
        np.testing.assert_allclose(forces[[0, 2]], -2*local*KJMOL_PER_EV)
        if step == 0:
            pid = p._session[0].process.pid
        assert p._session[0].process.pid == pid
    assert p._session[0].process.returncode == 0


@pytest.mark.parametrize(
    'change', ['qm_box', 'qm_origin', 'embedding_grid_shape'],
)
def test_persistent_rejects_mapping_changes(potential, change):
    p = potential
    object.__setattr__(p, 'qm_box', np.diag([10., 10., 6.]))
    object.__setattr__(p, 'qm_origin', np.array([0., 0., 1.]))
    object.__setattr__(p, 'embedding_grid_shape', (10, 10, 10))
    p.incar.update(NGXF=10, NGYF=10, NGZF=6)
    p.compute_energy()
    changes = dict(
        qm_box=np.diag([10., 10., 7.]), qm_origin=np.zeros(3),
        embedding_grid_shape=(10, 10, 20),
    )
    object.__setattr__(p, change, changes[change])
    p.system.positions[0, 0] += .1
    with pytest.raises(ValueError, match='changed'):
        p.compute_energy()
    assert p._session[0].process.poll() is not None


def test_dead_process_fails_instead_of_hanging(tmp_path):
    s = transport.PersistentVasp(
        f'{shlex.quote(sys.executable)} -c "raise SystemExit(7)"',
        tmp_path, dict(os.environ), 2, 5,
    )
    with pytest.raises(VaspExecutionError, match='code 7'):
        s.evaluate(np.zeros((1, 3)))
    assert s.closed


def test_timeout_reaps_process(tmp_path):
    s = transport.PersistentVasp(
        f'{shlex.quote(sys.executable)} -c "import time; time.sleep(30)"',
        tmp_path, dict(os.environ), 2, .1,
    )
    with pytest.raises(VaspExecutionError, match='Timed out'):
        s.evaluate(np.zeros((1, 3)))
    assert s.process.poll() is not None


def test_structure_refreshes_cache_even_without_geometry_motion(
        tmp_path, monkeypatch,
):
    monkeypatch.setenv(transport.SESSION_ENV, str(tmp_path))
    monkeypatch.setenv(transport.STEPS_ENV, '2')
    monkeypatch.setenv(transport.TIMEOUT_ENV, '1')
    monkeypatch.setattr(transport, '_plugin_frame', 0)
    p = np.array([[.1, .2, .3]])
    transport.publish(tmp_path/'request_000001.npz', step=1, positions=p)
    vasp_plugin._CACHE['external'] = 'previous MM sources'
    c = SimpleNamespace(positions=p, total_energy=1., forces=np.ones((1, 3)))
    a = SimpleNamespace(positions=np.zeros_like(p))
    transport.structure(c, a)
    assert vasp_plugin._CACHE == {}
    np.testing.assert_array_equal(a.positions, 0)
    with np.load(tmp_path/'result_000000.npz') as result:
        assert result['step'] == 0


def test_embedding_updates_without_qm_motion(potential):
    from dataclasses import replace
    p = replace(potential, embedding=True)
    p.potentials.append(
        SimpleNamespace(
            pme_alpha=.5, pme_gridnumber=(20, 20, 20), pme_spline_order=5,
        ),
    )
    p.system.subsystems[1] = Subsystem.II
    p.system.charges[1] = .4
    try:
        e0 = p.compute_energy()
        np.testing.assert_array_equal(p.compute_forces()[1], [1, 2, 3])
        p.system.positions[1, 0] += .2
        e1 = p.compute_energy()
        assert e1 != e0
        p.system.subsystems[1] = Subsystem.III
        e2 = p.compute_energy()
        assert e2 != e1
        # Subsystem III gets the plugin's reciprocal reaction.
        np.testing.assert_array_equal(p.compute_forces()[1], [4, 5, 6])
        assert p.frame[0] == 3 and p._session[0].closed
    finally:
        p.close()


def test_run_vasp_forwards_environment(tmp_path):
    from pydft_qmmm.interfaces.vasp.vasp_utils import run_vasp
    env = dict(os.environ, PYDFT_ENV_TEST='expected')
    script = "import os; assert os.environ['PYDFT_ENV_TEST']=='expected'"
    run_vasp(
        f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}',
        str(tmp_path), env,
    )


@pytest.mark.parametrize('missing_step', [False, True])
def test_persistent_convergence_check(potential, missing_step):
    p = potential
    object.__setattr__(p, 'check_convergence', True)
    if missing_step:
        engine = Path(p.directory).parent / 'fake_engine.py'
        source = engine.read_text().replace(
            "with open('OUTCAR', 'a') as fh:",
            "with open('OUTCAR', 'w') as fh:",
        )
        engine.write_text(source)
    for step in range(2):
        p.system.positions[0, 0] += .1
        p.compute_energy()
    p.system.positions[0, 0] += .1
    if missing_step:
        with pytest.raises(
                VaspExecutionError, match='Only 1/3 SCFs converged',
        ):
            p.compute_energy()
    else:
        assert np.isfinite(p.compute_energy())
    assert p._session[0].process.returncode == 0


def test_convergence_check_requires_outcar(potential):
    with pytest.raises(VaspExecutionError, match='Cannot read OUTCAR'):
        potential._check_convergence(1)
