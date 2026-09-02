import unittest

import pandas as pd

from simulation import (
    SimulationSettings,
    build_demand_scenario_table,
    run_experiment,
    sample_demand,
    validate_determined_stock,
    validate_products,
)


def base_product(**changes):
    row = {
        "Producto": "Prueba",
        "Stock_inicial": 100,
        "Distribucion_demanda": "Normal",
        "Demanda_media": 5,
        "Demanda_desv": 0,
        "Demanda_min": 2,
        "Demanda_moda": 5,
        "Demanda_max": 8,
        "Lead_time_media": 2,
        "Lead_time_desv": 0,
        "Costo_unitario": 10,
        "Costo_orden": 20,
        "Tasa_mantenimiento_anual": 0.20,
        "Costo_quiebre_unidad": 15,
        "Q": 30,
        "s": 15,
        "T": 5,
        "S": 40,
    }
    row.update(changes)
    return row


class SimulationTests(unittest.TestCase):
    def test_determined_stock_is_exact_with_deterministic_demand(self):
        row = pd.Series(base_product(Stock_inicial=50))
        settings = SimulationSettings(replications=100, protection_days=10, target_probability=0.95)
        summary, _ = validate_determined_stock(row, settings)
        self.assertEqual(summary["Stock_recomendado_objetivo"], 50)
        self.assertEqual(summary["Probabilidad_cobertura"], 1.0)
        self.assertEqual(summary["Evaluacion"], "Suficiente")
        self.assertGreaterEqual(summary["Demanda_CVaR95_periodo"], summary["Demanda_p95_periodo"])
        self.assertTrue(0 <= summary["Utilizacion_stock"] <= 1)

    def test_stock_gap_detects_shortage(self):
        row = pd.Series(base_product(Stock_inicial=49))
        settings = SimulationSettings(replications=100, protection_days=10, target_probability=0.95)
        summary, _ = validate_determined_stock(row, settings)
        self.assertEqual(summary["Probabilidad_cobertura"], 0.0)
        self.assertEqual(summary["Brecha_stock"], -1)
        self.assertEqual(summary["Evaluacion"], "Insuficiente")

    def test_scenario_detail_reconciles_with_stock_summary(self):
        row = pd.Series(base_product(Stock_inicial=50, Demanda_desv=1))
        settings = SimulationSettings(replications=250, protection_days=10, seed=77)
        summary, _ = validate_determined_stock(row, settings)
        scenarios = build_demand_scenario_table(row, settings)
        self.assertEqual(len(scenarios), settings.replications)
        self.assertAlmostEqual(float(scenarios["Cobertura"].mean()), summary["Probabilidad_cobertura"])
        self.assertTrue((scenarios["Faltante"] >= 0).all())
        self.assertTrue((scenarios["Excedente"] >= 0).all())

    def test_each_policy_returns_bounded_metrics(self):
        products = pd.DataFrame([base_product()])
        settings = SimulationSettings(horizon_days=30, replications=50, protection_days=10)
        results = run_experiment(products, ["Q,s", "T,S", "s,S"], settings)
        summary = results["comparacion_politicas"]
        self.assertEqual(len(summary), 3)
        self.assertTrue(summary["Nivel_servicio_unidades"].between(0, 1).all())
        self.assertTrue(summary["Dias_sin_quiebre"].between(0, 1).all())
        self.assertEqual(int(summary["Recomendada"].sum()), 1)
        self.assertEqual(len(results["escenarios_demanda"]), settings.replications)
        self.assertTrue(summary["Probabilidad_quiebre_horizonte"].between(0, 1).all())
        self.assertTrue(((summary["Costo_relevante_CVaR95"] + 1e-9) >= summary["Costo_relevante_VaR95"]).all())
        self.assertTrue(((summary["Costo_total_CVaR95"] + 1e-9) >= summary["Costo_total_p95"]).all())
        calculated = (
            summary["Costo_mantenimiento_promedio"]
            + summary["Costo_ordenamiento_promedio"]
            + summary["Costo_quiebre_promedio"]
        )
        self.assertTrue((calculated - summary["Costo_relevante_promedio"]).abs().lt(1e-8).all())

    def test_validation_rejects_s_below_s(self):
        products = pd.DataFrame([base_product(s=50, S=40)])
        errors = validate_products(products)
        self.assertTrue(any("S debe ser" in error for error in errors))

    def test_empirical_requires_history(self):
        products = pd.DataFrame([base_product(Distribucion_demanda="Empirica")])
        errors = validate_products(products)
        self.assertTrue(any("Demanda_historica" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
