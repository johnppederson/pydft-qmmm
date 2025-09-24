from __future__ import annotations

import pytest

from pydft_qmmm import QMMMHamiltonian
from pydft_qmmm.utils import numerical_gradient


class TestCutoffEmbeddingSchemes:

    def test_electrostatic_cutoff_electronic_gradients(
            self,
            spce_qmmm_system,
            mm_spce_no_lj,
            qm_water,
    ):
        qmmm = QMMMHamiltonian(
            "electrostatic",
            "cutoff",
            partition=None,
        )
        total = mm_spce_no_lj[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0}, components=["Psi4"])
        assert analytical - numerical == pytest.approx(0, abs=0.5)

    def test_mechanical_cutoff_electronic_gradients(
            self,
            spce_qmmm_system,
            mm_spce_no_lj,
            qm_water,
    ):
        qmmm = QMMMHamiltonian(
            "mechanical",
            "cutoff",
            partition=None,
        )
        total = mm_spce_no_lj[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0}, components=["Psi4"])
        assert analytical - numerical == pytest.approx(0, abs=0.5)

    def test_electrostatic_electrostatic_electronic_gradients(
            self,
            spce_qmmm_system,
            mm_spce_no_lj,
            qm_water,
    ):
        qmmm = QMMMHamiltonian(
            "electrostatic",
            "electrostatic",
            pme_gridnumber=60,
            partition=None,
        )
        total = mm_spce_no_lj[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(
            calculator,
            {0},
            components=["Psi4", "PMENuclear"],
        )
        assert analytical - numerical == pytest.approx(0, abs=0.5)


class TestConservativeEmbeddingSchemes:

    def test_none_none_gradients(
            self,
            spce_qmmm_system,
            mm_spce,
            qm_water,
    ):
        qmmm = QMMMHamiltonian("none", "none")
        total = mm_spce[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=1)

    def test_mechanical_none_gradients(
            self,
            spce_qmmm_system,
            mm_spce,
            qm_water,
    ):
        qmmm = QMMMHamiltonian("mechanical", "none")
        total = mm_spce[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=1)

    def test_mechanical_mechanical_gradients(
            self,
            spce_qmmm_system,
            mm_spce,
            qm_water,
    ):
        qmmm = QMMMHamiltonian("mechanical", "mechanical")
        total = mm_spce[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=1)

    def test_electrostatic_none_gradients(
            self,
            spce_qmmm_system,
            mm_spce,
            qm_water,
    ):
        qmmm = QMMMHamiltonian("electrostatic", "none")
        total = mm_spce[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=1)

    def test_electrostatic_mechanical_gradients(
            self,
            spce_qmmm_system,
            mm_spce,
            qm_water,
    ):
        qmmm = QMMMHamiltonian("electrostatic", "mechanical")
        total = mm_spce[3:] + qm_water[0:3] + qmmm
        calculator = total.build_calculator(spce_qmmm_system)
        analytical = -calculator.calculate().forces[0]
        numerical = numerical_gradient(calculator, {0})
        assert analytical - numerical == pytest.approx(0, abs=1)
