"""Compare restart and persistent VASP on three embedded-water geometries.

Run on a compute node with PYDFT_QMMM_VASP_COMMAND and VASP_PP_PATH set.
The second frame moves only MM sources; the third also moves QM atoms and
changes the near/far partition. All energies and forces must match restart
calculations before interpreting timings. This is a correctness check, not a
production benchmark.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import time
import numpy as np
from pydft_qmmm import Atom, System
from pydft_qmmm.interfaces.vasp.vasp_factory import vasp_interface_factory
from pydft_qmmm.utils import Subsystem

root=Path(os.environ.get('CHECK_DIR','persistent_check')).resolve()
root.mkdir(parents=True,exist_ok=True)
water=np.array([[0.,0.,0.],[.9572,0.,0.],[-.239987,.926627,0.]])
initial=np.concatenate([water+[3,3,3],water+[6,3,3],water+[9,8,8]])
frames=[]
for step in range(3):
    xyz=initial.copy()
    if step>=1:xyz[3:6]+=[.2,.1,0]
    if step>=2:xyz[:3]+=[.015,-.01,.02]
    frames.append(xyz)
records={}
for mode in ('restart','persistent_0','persistent_11','persistent_12'):
    system=System([Atom(element=e,position=p,charge=q,subsystem=sub)
        for m in range(3) for e,p,q,sub in zip(('O','H','H'),initial[3*m:3*m+3],
             (-.8476,.4238,.4238),[Subsystem.I if m==0 else Subsystem.II if m==1 else Subsystem.III]*3)],
        box=np.eye(3)*14.)
    tags=dict(ENCUT=350,EDIFF=1e-7,NELM=100,ISYM=0,LREAL=False,
              NGXF=96,NGYF=96,NGZF=96,LWAVE=True,LCHARG=False,NELMIN=4,
              PREC='Normal',GGA='PE',ISMEAR=0,SIGMA=.05)
    if mode!='restart':tags['IWAVPR']=int(mode.split('_')[1])
    potential=vasp_interface_factory(system,directory=str(root/mode),embedding=True,
              incar=tags,persistent=mode!='restart',persistent_steps=len(frames),persistent_timeout=300)
    potential.potentials.append(SimpleNamespace(pme_alpha=.5,pme_gridnumber=(32,32,32),pme_spline_order=5))
    values=[]
    try:
        for step,xyz in enumerate(frames):
            system.positions=xyz
            if step==2:
                system.subsystems[3:6]=Subsystem.III
                system.subsystems[6:9]=Subsystem.II
            if step and mode=='restart':potential.incar.update(ISTART=1,ICHARG=0)
            start=time.perf_counter()
            energy=potential.compute_energy();forces=potential.compute_forces()
            values.append(dict(step=step,energy_kj_mol=energy,forces=forces.tolist(),seconds=time.perf_counter()-start))
            print(mode,step,energy,values[-1]['seconds'],flush=True)
    finally:
        potential.close()
    records[mode]=values
    (root/'results.json').write_text(json.dumps(records,indent=2)+'\n')
comparisons={}
for mode,values in records.items():
    if mode=='restart':continue
    de=max(abs(a['energy_kj_mol']-b['energy_kj_mol']) for a,b in zip(values,records['restart']))
    df=max(np.max(np.abs(np.array(a['forces'])-b['forces'])) for a,b in zip(values,records['restart']))
    comparisons[mode]=dict(max_energy_difference_kj_mol=de,max_force_difference_kj_mol_A=float(df))
    assert de<.02 and df<.1,(mode,de,df)
(root/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
print(json.dumps(comparisons,indent=2),flush=True)
