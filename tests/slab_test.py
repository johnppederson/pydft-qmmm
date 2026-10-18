"""Energy derivatives and guards for neutral slab coupling."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pydft_qmmm.embedding.slab import coefficient, mm_correction, qm_dipole, neutral_dipole
from pydft_qmmm.embedding.grid_potential import KJMOL_PER_EV


def test_mm_derivatives_and_translation():
    cell=np.diag([8.,9.,24.]);p=np.array([[2.,3.,-.4],[3.,4.,5.2],[1.,2.,7.]])
    q=np.array([-.8,.4,.4]);e,f=mm_correction(p,q,cell)
    for i in range(3):
        for axis in range(3):
            a,b=p.copy(),p.copy();a[i,axis]+=1e-5;b[i,axis]-=1e-5
            fd=-(mm_correction(a,q,cell)[0]-mm_correction(b,q,cell)[0])/2e-5
            assert f[i,axis]==pytest.approx(fd,abs=1e-9)
    assert mm_correction(p+[1,2,6],q,cell)[0]==pytest.approx(e)
    np.testing.assert_allclose(f.sum(axis=0),0,atol=1e-14)
    assert mm_correction(p,q,cell*2)[0]==pytest.approx(e/8)


def test_guards():
    with pytest.raises(ValueError,match='neutral'):
        mm_correction(np.zeros((1,3)),[1.],np.eye(3))
    with pytest.raises(ValueError,match='orthorhombic'):
        coefficient([[8,1,0],[0,8,0],[0,0,24]])
    with pytest.raises(ValueError,match='neutral QM'):
        qm_dipole(np.zeros((4,4,4)),np.zeros((1,3)),[1.],np.eye(3),[0,0,0])


@pytest.mark.parametrize("qm_self", [False, True])
@pytest.mark.parametrize("mm_height", [12., 28.])
def test_plugin_cross_derivatives(tmp_path,monkeypatch,qm_self,mm_height):
    from pydft_qmmm.interfaces.vasp import vasp_plugin as plugin
    from pydft_qmmm.interfaces.vasp.vasp_utils import write_mm_charges,read_mm_forces
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plugin,'_external_potential',lambda c:np.zeros(tuple(c.shape_grid)))
    monkeypatch.setattr(plugin,'contract_gaussian_gradient',lambda *args:np.zeros((2,3)))
    monkeypatch.delenv(plugin.ERFC_NEAR_ENV,raising=False)
    monkeypatch.delenv(plugin.COARSE_NEAR_ENV,raising=False)
    cell=np.diag([8.,8.,12.]);shape=(8,8,12);origin=[0,0,-2.]
    density=np.zeros(shape);density[2,2,3]=np.prod(shape)*.7;density[3,2,4]=np.prod(shape)*1.3
    ions=np.array([[2.1,2.,2.5],[3.1,2.,5.2]])
    mm=np.array([[2.,2.,1.1],[4.,4.,8.4]]);q=np.array([.5,-.5])
    c=coefficient(np.diag([8.,8.,mm_height]));cq=coefficient(cell)
    def evaluate(x,y,rho=density):
        write_mm_charges('MM_CHARGES',y,q,7,.5)
        (tmp_path/'SLAB_DATA.json').write_text(json.dumps(dict(coefficient=c,qm_coefficient=cq,
            moment=neutral_dipole(y,q),charges=q.tolist(),origin=origin,step=7,qm_self=qm_self)))
        const=SimpleNamespace(lattice_vectors=cell,shape_grid=shape,positions=x@np.linalg.inv(cell),
            ZVAL=np.array([1.]),ion_types=np.zeros(2,dtype=int),charge_density=rho,forces=np.zeros((2,3)))
        local=SimpleNamespace(total_potential=np.zeros(shape),total_energy=0.);plugin.local_potential(const,local)
        additions=SimpleNamespace(total_energy=0.,forces=np.zeros((2,3)));plugin.force_and_stress(const,additions)
        energy=local.total_energy+additions.total_energy+np.sum(rho/np.prod(shape)*local.total_potential)
        reaction=read_mm_forces('SLAB_FORCES',7)[0]/KJMOL_PER_EV
        return energy,additions.forces,reaction,local.total_potential
    energy,fi,fm,v=evaluate(ions,mm)
    mq=qm_dipole(density,ions,[1.,1.],cell,origin)
    assert energy==pytest.approx(2*c*neutral_dipole(mm,q)*mq + (cq*mq**2 if qm_self else 0.))
    for which in [0,1]:
        for i in range(2):
            a=[ions.copy(),mm.copy()];b=[ions.copy(),mm.copy()]
            a[which][i,2]+=1e-5;b[which][i,2]-=1e-5
            fd=-(evaluate(*a)[0]-evaluate(*b)[0])/2e-5
            assert [fi,fm][which][i,2]==pytest.approx(fd,abs=1e-8)
    a,b=density.copy(),density.copy();h=1e-5
    for rho,sign in [(a,1),(b,-1)]:
        rho[2,2,3]+=sign*h*np.prod(shape);rho[3,2,4]-=sign*h*np.prod(shape)
    derivative=(evaluate(ions,mm,a)[0]-evaluate(ions,mm,b)[0])/(2*h)
    assert derivative==pytest.approx(v[2,2,3]-v[3,2,4],abs=1e-9)
    assert qm_dipole(density,ions,[1.,1.],cell,[0,0,3.])==pytest.approx(mq)
    plugin.reset_cache()


def test_openmm_excluded_charges(spce_system,mm_spce):
    potential=mm_spce.build_calculator(spce_system).potential
    potential.zero_charges(frozenset([0,1,2]))
    energy=potential.compute_energy();forces=potential.compute_forces()
    object.__setattr__(potential,'slab_correction',True)
    q=np.array(spce_system.charges);q[:3]=0
    e,f=mm_correction(spce_system.positions,q,spce_system.box)
    assert potential.compute_energy()-energy==pytest.approx(e*KJMOL_PER_EV,abs=1e-7)
    np.testing.assert_allclose(potential.compute_forces()-forces,f*KJMOL_PER_EV,atol=1e-7)
    assert potential.compute_components()['Slab MM Energy']==pytest.approx(e*KJMOL_PER_EV)


def test_driver_adds_reaction_once(tmp_path,monkeypatch):
    from pydft_qmmm import Atom,System
    from pydft_qmmm.utils import Subsystem
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    from pydft_qmmm.interfaces.vasp.vasp_interface import VaspPotential
    system=System([Atom(element='H',subsystem=r) for r in
                   [Subsystem.I]*2+[Subsystem.II]*2+[Subsystem.III]*2],box=np.eye(3)*20)
    p=vasp_interface_factory(system,pp_path='unused',directory=str(tmp_path),embedding=True)
    object.__setattr__(p,'slab_correction',True);p.potentials.append(object());p.frame[0]=1
    monkeypatch.setattr(VaspPotential,'_run',lambda self:(0.,np.zeros((2,3))))
    monkeypatch.setattr(VaspPotential,'_read_qm_net_force',lambda self:np.zeros(3))
    def write(name,values):
        with (tmp_path/name).open('w') as f:
            f.write(f'{len(values)} 0\n');np.savetxt(f,values)
    write('MM_FORCES',np.ones((2,3)))
    write('PME_FORCES',np.full((6,3),2.))
    reaction=np.zeros((6,3));reaction[2:,2]=3.
    write('SLAB_FORCES',reaction)
    expected=np.zeros((6,3));expected[2:4]=1.;expected[4:]=2.;expected+=reaction
    np.testing.assert_allclose(p.compute_forces(),expected)
    write('SLAB_FORCES',np.zeros((2,3)))
    with pytest.raises(ValueError,match='shape'):p.compute_forces()


def test_public_option_rejects_unsupported_coupling():
    from pydft_qmmm import QMMMHamiltonian
    for kw in [{},{'partition':None,'long_range':'cutoff'},
               {'partition':None,'long_range':'electrostatic','coupling_mode':'force'}]:
        with pytest.raises(ValueError,match='Slab correction'):
            QMMMHamiltonian(slab_correction=True,**kw)


def test_qm_small_grid_residual_is_not_renormalized():
    shape=(4,4,8);cell=np.diag([4.,4.,8.]);origin=np.array([0.,0.,-2.])
    ions=np.array([[1.,1.,2.],[2.,1.,3.]])
    density=np.zeros(shape);density[1,1,3]=(2.+2.1591e-5)*np.prod(shape)
    original=density.copy()
    moment=qm_dipole(density,ions,[1.,1.],cell,origin)
    expected=(ions[:,2]+origin[2]).sum()-(2.+2.1591e-5)*(3.+origin[2])
    assert moment==pytest.approx(expected,abs=1e-12)
    np.testing.assert_array_equal(density,original)
    density[1,1,3]=(2.+1e-3)*np.prod(shape)
    with pytest.raises(ValueError,match='neutral QM'):
        qm_dipole(density,ions,[1.,1.],cell,origin)


def test_qm_self_requires_full_slab_correction():
    from pydft_qmmm import QMMMHamiltonian
    with pytest.raises(ValueError, match='requires slab_correction'):
        QMMMHamiltonian(qm_slab_correction=True)


@pytest.mark.parametrize('condition', ['valid', 'truncated', 'short', 'ldipol'])
def test_periodic_self_cell_guards(tmp_path, condition):
    from pydft_qmmm import Atom, System
    from pydft_qmmm.utils import Subsystem
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    pp=tmp_path/'pp'/'H';pp.mkdir(parents=True);(pp/'POTCAR').write_text('test\n')
    cell=np.diag([8.,8.,16.]);qm=cell.copy()
    if condition=='short':qm[2,2]=8.
    system=System([Atom(element='H',position=np.array([1.,2.,1.]),subsystem=Subsystem.I)],box=cell)
    tags=dict(NGXF=8,NGYF=8,NGZF=int(qm[2,2]),LDIPOL=condition=='ldipol')
    if condition=='truncated':tags['KERNEL_TRUNCATION/LTRUNCATE']=True
    potential=vasp_interface_factory(system,pp_path=str(pp.parent),directory=str(tmp_path/'work'),
                                    embedding=True,qm_box=qm,qm_origin=[0,0,-2],
                                    embedding_grid_shape=(8,8,16),incar=tags)
    object.__setattr__(potential,'slab_correction',True)
    object.__setattr__(potential,'qm_slab_correction',True)
    if condition in ['valid','short']:
        potential._write_input()
        metadata=json.loads((tmp_path/'work/SLAB_DATA.json').read_text())
        assert metadata['qm_self'] is True
        assert metadata['qm_coefficient']==pytest.approx(coefficient(qm))
        assert metadata['coefficient']==pytest.approx(coefficient(cell))
    else:
        with pytest.raises(ValueError,match='QM slab correction'):
            potential._write_input()


def test_qm_neutrality_allowance_scales_with_electron_count():
    # 4000 valence electrons: 1e-7 per electron allows 4e-4 e, above the 1e-4 floor.
    shape=(4,4,8);cell=np.diag([4.,4.,8.]);origin=np.zeros(3)
    ions=np.array([[1.,1.,2.],[2.,1.,3.]]);valence=[2000.,2000.]
    density=np.zeros(shape);density[1,1,2]=(4000.+2.3e-4)*np.prod(shape)
    qm_dipole(density,ions,valence,cell,origin)
    density[1,1,2]=(4000.+1e-3)*np.prod(shape)
    with pytest.raises(ValueError,match='neutral QM'):
        qm_dipole(density,ions,valence,cell,origin)


@pytest.mark.parametrize('condition', ['valid', 'idipol', 'dipol', 'truncated'])
def test_vasp_ldipol_replaces_qm_self_term(tmp_path, condition):
    from pydft_qmmm import Atom, System
    from pydft_qmmm.utils import Subsystem
    from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
    pp=tmp_path/'pp'/'H';pp.mkdir(parents=True);(pp/'POTCAR').write_text('test\n')
    cell=np.diag([8.,8.,16.])
    system=System([Atom(element='H',position=np.array([1.,2.,1.]),subsystem=Subsystem.I)],box=cell)
    tags=dict(NGXF=8,NGYF=8,NGZF=16,LDIPOL=True,IDIPOL=3,DIPOL=[0.5,0.5,0.5])
    if condition=='idipol':tags['IDIPOL']=4
    if condition=='dipol':tags['DIPOL']=[0.5,0.5,0.3]
    if condition=='truncated':tags['KERNEL_TRUNCATION/LTRUNCATE']=True
    potential=vasp_interface_factory(system,pp_path=str(pp.parent),directory=str(tmp_path/'work'),
                                    embedding=True,qm_box=cell,qm_origin=[0,0,-2],
                                    embedding_grid_shape=(8,8,16),incar=tags)
    object.__setattr__(potential,'slab_correction',True)
    if condition=='valid':
        potential._write_input()
        assert json.loads((tmp_path/'work/SLAB_DATA.json').read_text())['qm_self'] is False
    else:
        with pytest.raises(ValueError,match='LDIPOL slab correction'):
            potential._write_input()
