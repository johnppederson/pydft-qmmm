"""Embedding configuration contracts for VASP and SPARC; no engine runs."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pydft_qmmm.interfaces import ElectrostaticCouplingMode
from pydft_qmmm.interfaces.sparc.sparc_interface import SPARCInterface
from pydft_qmmm.interfaces.vasp.vasp_interface import VaspInterface


@pytest.fixture(params=["vasp", "sparc"])
def adapter(request, tmp_path):
    common = dict(system=None, charge=0, directory=str(tmp_path))
    if request.param == "vasp":
        return (
            lambda embedding: VaspInterface(
                **common, command="", incar={}, kpts=(1, 1, 1),
                pp_path="", potcar_map={}, embedding=embedding,
            ),
            "VASP embedding=True conflicts with this QMMMHamiltonian: "
            "no QM/MM electrostatic interaction is assigned to the QM "
            "level. Disable embedding or select electrostatic coupling "
            "to avoid double-counting electrostatics.",
            "PME embedding needs the VASP Python plugin.  Build the "
            "potential with embedding=True and a vasp_std compiled "
            "with -DPLUGINS.",
        )
    return (
        lambda embedding: SPARCInterface(
            **common, calculator=None, fd_grid=(8, 8, 8),
            embedding=embedding,
        ),
        "SPARC embedding=True conflicts with this "
        "QMMMHamiltonian: no QM/MM electrostatic interaction is "
        "assigned to the QM level of theory.  Disable embedding "
        "or select electrostatic coupling to avoid "
        "double-counting electrostatics.",
        "PME embedding needs the SPARC QM/MM fork.  Build the "
        "potential with embedding=True and point the command "
        "keyword at a sparc built from the qmmm-embedding "
        "branch.",
    )


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_configuration_preserves_state_and_nuclear_ownership(
        adapter, initial, enabled,
):
    factory, conflict, _ = adapter
    interface = factory(initial)
    if initial and not enabled:
        with pytest.raises(ValueError) as error:
            interface.configure_electrostatic_embedding(enabled)
        assert str(error.value) == conflict
    else:
        assert interface.configure_electrostatic_embedding(enabled) is None
    expected = initial or enabled
    assert interface.embedding is expected
    assert interface.applies_nuclear_potential() is expected
    assert interface.electrostatic_coupling_mode() is ElectrostaticCouplingMode.ENGINE
    assert interface.potentials == []
    with pytest.raises(FrozenInstanceError):
        interface.embedding = False


def test_registration_requires_embedding_and_preserves_order(adapter):
    factory, conflict, unavailable = adapter
    interface = factory(False)
    first, second = object(), object()
    with pytest.raises(NotImplementedError) as error:
        interface.add_electronic_potential(first)
    assert str(error.value) == unavailable
    assert interface.potentials == []

    interface.configure_electrostatic_embedding(True)
    for potential in (first, second, first):
        assert interface.add_electronic_potential(potential) is None
    interface.configure_electrostatic_embedding(True)
    with pytest.raises(ValueError) as error:
        interface.configure_electrostatic_embedding(False)
    assert str(error.value) == conflict
    assert interface.embedding is True
    assert interface.potentials == [first, second, first]
    assert factory(True).potentials == []
