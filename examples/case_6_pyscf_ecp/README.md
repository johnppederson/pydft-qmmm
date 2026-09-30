Example Case 6 (pyscf-mol with ECP)
==================================

Compute an iodide ion's energy and force in SPC/E water using
`interface="pyscf-mol"`, PBE, `basis="def2-svp"`, and `ecp="def2-svp"`.
Region II contains waters within 8 Å of the ion; region III enters
through PME.

Set both `basis` and `ecp`: the valence basis alone cannot represent
all electrons. The interface uses the ECP valence charge in nuclear
embedding.

Run from this directory on a compute node with PySCF and helPME-py:

```bash
python case_6_ecp.py
```

The script prints **26 explicit electrons** and a **nuclear valence
charge of 25**, followed by energies and the ion's force. Without the
ECP, these counts would be 54 and 53. Energy components retain the names
`PySCF`, `OpenMM`, and `PMENuclear`.

For periodic QM calculations, `pyscf-pbc` requires a GTH `pseudo`
instead of an ECP; see [case 8](../case_8_pyscf_pbc).
