"""Conservation checks for aligned, separate QM and MM cells."""
import numpy as np
import pytest
from pydft_qmmm.embedding.cell_mapping import CellMapping
from pydft_qmmm.embedding.grid_potential import (
    build_external_potential, poisson_fft, contract_gaussian_gradient,
)


def mapping():
    return CellMapping(np.diag([8., 8., 16.]), np.diag([8., 8., 8.]),
                       [0., 0., -2.], (24, 24, 48), (24, 24, 24))


def test_restriction_transpose():
    m = mapping()
    rng = np.random.default_rng(5)
    v = rng.normal(size=m.mm_shape)
    q = rng.normal(size=m.qm_shape)
    assert np.sum(q*m.restrict(v)) == pytest.approx(np.sum(m.transpose(q)*v), abs=1e-10)
    assert m.transpose(q).sum() == pytest.approx(q.sum())


@pytest.mark.parametrize('source_z', [5.2, 6.0, 9.2])
def test_gaussian_force_derivative(source_z):
    m = mapping()
    rng = np.random.default_rng(8)
    q = rng.normal(size=m.qm_shape)*1e-3
    q -= q.mean()
    # The second source may lie outside the cropped QM cell (z=-2..6).
    positions = np.array([[3.2, 4.1, 1.7], [2.8, 3.7, source_z]])
    charges = np.array([0.7, -0.7])
    rho = m.transpose(q)*np.prod(m.mm_shape)/np.linalg.det(m.mm_cell)
    phi = poisson_fft(rho, m.mm_cell)
    f = contract_gaussian_gradient(phi, positions, charges, m.mm_shape, m.mm_cell, .6)
    def energy(p):
        # build_external_potential is electron potential energy, -phi.
        return -np.sum(q*m.restrict(build_external_potential(p, charges, m.mm_shape, m.mm_cell, .6)))
    for i in range(2):
        for axis in range(3):
            plus, minus = positions.copy(), positions.copy()
            plus[i,axis] += 1e-5
            minus[i,axis] -= 1e-5
            assert f[i,axis] == pytest.approx(-(energy(plus)-energy(minus))/2e-5, abs=1e-7)


def test_geometry_validation():
    m=mapping()
    m.validate_positions([[0., 0., 0.]])
    with pytest.raises(ValueError): m.validate_positions([[0., 0., 6.]])
    with pytest.raises(ValueError):
        CellMapping(m.mm_cell, m.qm_cell, [0,0,.1], m.mm_shape, m.qm_shape)
    with pytest.raises(ValueError):
        CellMapping(m.mm_cell, m.qm_cell, m.origin, m.mm_shape, (24,24,25))


def test_pme_separate_lattice_force(tmp_path):
    pytest.importorskip('helpme_py')
    from pydft_qmmm.embedding.pme_grid import write_pme_data, build_pme_potential, pme_source_forces, KJMOL_PER_EV
    m=mapping()
    p=np.array([[1.,2.,10.], [4.,3.,12.]])
    c=np.array([.5,-.5]); path=str(tmp_path/'PME_DATA')
    rng=np.random.default_rng(10); q=rng.normal(size=m.qm_shape)*.001
    def write(x): write_pme_data(path,x,c,[],.5,(16,16,32),6,0)
    kw=dict(source_cell=m.mm_cell,origin=m.origin)
    write(p)
    f=pme_source_forces(path,q,m.qm_shape,m.qm_cell,**kw)/KJMOL_PER_EV
    def energy(x):
        write(x)
        return -np.sum(q*build_pme_potential(path,m.qm_shape,m.qm_cell,**kw))
    for i in range(2):
        a,b=p.copy(),p.copy(); a[i,2]+=1e-4; b[i,2]-=1e-4
        assert f[i,2] == pytest.approx(-(energy(a)-energy(b))/2e-4,abs=2e-6)


def test_equal_cell_is_identity():
    m=CellMapping(np.diag([8.,8.,16.]),np.diag([8.,8.,16.]),[0,0,0],(12,12,24),(12,12,24))
    q=np.random.default_rng(1).normal(size=m.mm_shape)
    np.testing.assert_array_equal(m.restrict(q),q)
    np.testing.assert_array_equal(m.transpose(q),q)


def test_plugin_mapping_and_cache(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from pydft_qmmm.interfaces.vasp import vasp_plugin as plugin
    from pydft_qmmm.interfaces.vasp.vasp_utils import write_mm_charges
    from pydft_qmmm.embedding.cell_mapping import read_mapping
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(plugin.ERFC_NEAR_ENV,raising=False)
    monkeypatch.delenv(plugin.COARSE_NEAR_ENV,raising=False)
    m=mapping()
    (tmp_path/'QM_CELL.json').write_text(json.dumps(m.metadata()))
    p=np.array([[3.,4.,1.], [3.,4.,5.]])
    c=np.array([.5,-.5])
    write_mm_charges('MM_CHARGES',p,c,0,.6)
    plugin.reset_cache()
    constants=SimpleNamespace(shape_grid=m.qm_shape,lattice_vectors=m.qm_cell)
    actual=plugin._external_potential(constants)
    expected=m.restrict(build_external_potential(p,c,m.mm_shape,m.mm_cell,.6))
    np.testing.assert_allclose(actual,expected)
    changed=CellMapping(m.mm_cell,m.qm_cell,[0,0,-1.],m.mm_shape,m.qm_shape)
    (tmp_path/'QM_CELL.json').write_text(json.dumps(changed.metadata()))
    np.testing.assert_allclose(plugin._external_potential(constants),changed.restrict(
        build_external_potential(p,c,m.mm_shape,m.mm_cell,.6)))
    with pytest.raises(ValueError):read_mapping((24,24,25),m.qm_cell)
    plugin.reset_cache()


def test_interface_keeps_global_system_and_writes_local_poscar(tmp_path):
    from pydft_qmmm import Atom, System
    from pydft_qmmm.utils import Subsystem
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    import json
    m=mapping()
    pp=tmp_path/'pp'/'H'; pp.mkdir(parents=True);(pp/'POTCAR').write_text('test\n')
    system=System([Atom(element='H',position=np.array([1.,2.,1.]),subsystem=Subsystem.I),
                   Atom(element='H',position=np.array([2.,3.,2.]),subsystem=Subsystem.II)],box=m.mm_cell.T)
    positions=np.array(system.positions,copy=True)
    kwargs=dict(pp_path=str(pp.parent),directory=str(tmp_path/'work'),qm_box=m.qm_cell.T,
                qm_origin=m.origin,embedding_grid_shape=m.mm_shape,
                incar=dict(NGXF=24,NGYF=24,NGZF=24))
    p=vasp_interface_factory(system,**kwargs)
    p._write_input()
    lines=(tmp_path/'work'/'POSCAR').read_text().splitlines()
    np.testing.assert_allclose([float(x) for x in lines[8].split()],positions[0]-m.origin)
    np.testing.assert_array_equal(system.positions,positions)
    np.testing.assert_array_equal(system.box,m.mm_cell.T)
    assert json.loads((tmp_path/'work'/'QM_CELL.json').read_text())['mm_shape']==list(m.mm_shape)
    persistent = vasp_interface_factory(system,persistent=True,**kwargs)
    persistent._write_input()
    assert 'IBRION = 12' in (tmp_path/'work'/'INCAR').read_text()
    system.positions[1,2]=6.
    p._write_input()  # Region II sources may cross the QM faces.
    system.positions[0,2]=6.
    with pytest.raises(ValueError,match='QM atoms'):p._write_input()
