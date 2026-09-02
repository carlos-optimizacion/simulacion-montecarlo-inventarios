"""Motor de simulacion Monte Carlo para politicas de inventario.

El modulo no depende de Streamlit, por lo que puede reutilizarse en notebooks,
pruebas o servicios web.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import hashlib

import numpy as np
import pandas as pd


POLICY_LABELS = {
    "Q,s": "(Q,s) Cantidad fija y punto de pedido",
    "T,S": "(T,S) Revision periodica hasta S",
    "s,S": "(s,S) Minimo-maximo",
}


REQUIRED_PRODUCT_COLUMNS = [
    "Producto",
    "Stock_inicial",
    "Distribucion_demanda",
    "Demanda_media",
    "Demanda_desv",
    "Demanda_min",
    "Demanda_moda",
    "Demanda_max",
    "Lead_time_media",
    "Lead_time_desv",
    "Costo_unitario",
    "Costo_orden",
    "Tasa_mantenimiento_anual",
    "Costo_quiebre_unidad",
    "Q",
    "s",
    "T",
    "S",
]


@dataclass(frozen=True)
class SimulationSettings:
    horizon_days: int = 180
    replications: int = 1000
    protection_days: int = 30
    target_probability: float = 0.95
    seed: int = 2026


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(p) for p in parts)]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def normalize_distribution(value: object) -> str:
    raw = str(value).strip().lower()
    aliases = {
        "normal": "Normal",
        "poisson": "Poisson",
        "triangular": "Triangular",
        "empirica": "Empirica",
        "empírica": "Empirica",
        "historica": "Empirica",
        "histórica": "Empirica",
    }
    if raw not in aliases:
        raise ValueError(
            f"Distribucion '{value}' no reconocida. Use Normal, Poisson, "
            "Triangular o Empirica."
        )
    return aliases[raw]


def validate_products(
    products: pd.DataFrame,
    historical: pd.DataFrame | None = None,
) -> list[str]:
    errors: list[str] = []
    missing = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in products.columns]
    if missing:
        return ["Faltan columnas obligatorias: " + ", ".join(missing)]

    if products.empty:
        return ["Debe ingresar al menos un producto."]

    if products["Producto"].astype(str).str.strip().duplicated().any():
        errors.append("Los nombres de producto no pueden repetirse.")

    numeric_columns = [c for c in REQUIRED_PRODUCT_COLUMNS if c not in {"Producto", "Distribucion_demanda"}]
    for col in numeric_columns:
        converted = pd.to_numeric(products[col], errors="coerce")
        if converted.isna().any():
            errors.append(f"La columna {col} contiene valores no numericos o vacios.")

    if errors:
        return errors

    nonnegative = [
        "Stock_inicial", "Demanda_media", "Demanda_desv", "Demanda_min",
        "Demanda_moda", "Demanda_max", "Lead_time_desv", "Costo_unitario",
        "Costo_orden", "Tasa_mantenimiento_anual", "Costo_quiebre_unidad",
        "Q", "s", "T", "S",
    ]
    for col in nonnegative:
        if (pd.to_numeric(products[col]) < 0).any():
            errors.append(f"La columna {col} no admite valores negativos.")

    if (pd.to_numeric(products["Lead_time_media"]) < 1).any():
        errors.append("Lead_time_media debe ser como minimo 1 dia.")
    if (pd.to_numeric(products["T"]) < 1).any():
        errors.append("T debe ser como minimo 1 dia.")
    if (pd.to_numeric(products["Q"]) <= 0).any():
        errors.append("Q debe ser mayor que cero.")
    if (pd.to_numeric(products["S"]) < pd.to_numeric(products["s"])).any():
        errors.append("En la politica (s,S), S debe ser mayor o igual que s.")

    for _, row in products.iterrows():
        name = str(row["Producto"]).strip()
        try:
            distribution = normalize_distribution(row["Distribucion_demanda"])
        except ValueError as exc:
            errors.append(f"{name}: {exc}")
            continue
        if distribution == "Triangular":
            low = float(row["Demanda_min"])
            mode = float(row["Demanda_moda"])
            high = float(row["Demanda_max"])
            if not (low <= mode <= high) or low == high:
                errors.append(f"{name}: para Triangular debe cumplirse minimo <= moda <= maximo y minimo < maximo.")
        if distribution == "Empirica":
            if historical is None or historical.empty:
                errors.append(f"{name}: la distribucion Empirica requiere la hoja Demanda_historica.")
            elif not {"Producto", "Demanda"}.issubset(historical.columns):
                errors.append("Demanda_historica debe contener las columnas Producto y Demanda.")
            else:
                values = pd.to_numeric(
                    historical.loc[historical["Producto"].astype(str).str.strip() == name, "Demanda"],
                    errors="coerce",
                ).dropna()
                if values.empty:
                    errors.append(f"{name}: no tiene observaciones en Demanda_historica.")
                elif (values < 0).any():
                    errors.append(f"{name}: la demanda historica no puede ser negativa.")
    return errors


def _historical_values(product: str, historical: pd.DataFrame | None) -> np.ndarray | None:
    if historical is None or historical.empty:
        return None
    mask = historical["Producto"].astype(str).str.strip() == product
    values = pd.to_numeric(historical.loc[mask, "Demanda"], errors="coerce").dropna()
    return values.to_numpy(dtype=float) if not values.empty else None


def sample_demand(
    row: Mapping[str, object],
    rng: np.random.Generator,
    size: int | tuple[int, ...],
    historical_values: Sequence[float] | None = None,
) -> np.ndarray:
    distribution = normalize_distribution(row["Distribucion_demanda"])
    if distribution == "Normal":
        values = rng.normal(float(row["Demanda_media"]), float(row["Demanda_desv"]), size=size)
    elif distribution == "Poisson":
        values = rng.poisson(max(0.0, float(row["Demanda_media"])), size=size)
    elif distribution == "Triangular":
        values = rng.triangular(
            float(row["Demanda_min"]),
            float(row["Demanda_moda"]),
            float(row["Demanda_max"]),
            size=size,
        )
    else:
        if historical_values is None or len(historical_values) == 0:
            raise ValueError("La distribucion Empirica requiere demanda historica.")
        values = rng.choice(np.asarray(historical_values, dtype=float), size=size, replace=True)
    return np.maximum(0, np.rint(values)).astype(int)


def sample_lead_time(row: Mapping[str, object], rng: np.random.Generator) -> int:
    mean = float(row["Lead_time_media"])
    std = float(row["Lead_time_desv"])
    value = mean if std == 0 else rng.normal(mean, std)
    return max(1, int(round(value)))


def validate_determined_stock(
    row: Mapping[str, object],
    settings: SimulationSettings,
    historical: pd.DataFrame | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Evalua el stock inicial contra demanda acumulada sin reabastecimiento."""
    product = str(row["Producto"]).strip()
    rng = np.random.default_rng(_stable_seed(settings.seed, product, "stock"))
    history = _historical_values(product, historical)
    demand = sample_demand(
        row,
        rng,
        (settings.replications, settings.protection_days),
        history,
    )
    cumulative = demand.cumsum(axis=1)
    total = cumulative[:, -1]
    stock = int(round(float(row["Stock_inicial"])))
    shortage = np.maximum(total - stock, 0)
    served = total - shortage
    recommended = int(np.ceil(np.quantile(total, settings.target_probability, method="higher")))
    probability = float(np.mean(total <= stock))
    fill_rate = float(served.sum() / total.sum()) if total.sum() else 1.0
    first_stockout = np.where(cumulative > stock, np.arange(1, settings.protection_days + 1), settings.protection_days + 1)
    first_stockout = first_stockout.min(axis=1)
    stockout_mask = first_stockout <= settings.protection_days
    average_stockout_day = float(first_stockout[stockout_mask].mean()) if stockout_mask.any() else np.nan

    assessment = "Suficiente" if probability >= settings.target_probability else "Insuficiente"
    summary: dict[str, float | str] = {
        "Producto": product,
        "Stock_evaluado": stock,
        "Probabilidad_cobertura": probability,
        "Fill_rate_sin_reposicion": fill_rate,
        "Demanda_promedio_periodo": float(total.mean()),
        "Demanda_p95_periodo": float(np.quantile(total, 0.95)),
        "Stock_recomendado_objetivo": recommended,
        "Brecha_stock": stock - recommended,
        "Dia_promedio_quiebre": average_stockout_day,
        "Evaluacion": assessment,
    }
    percentile_table = pd.DataFrame({
        "Percentil": [50, 75, 80, 85, 90, 95, 97.5, 99],
        "Demanda_acumulada": [
            int(np.ceil(np.quantile(total, q, method="higher")))
            for q in [0.50, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
        ],
    })
    percentile_table.insert(0, "Producto", product)
    return summary, percentile_table


def _order_quantity(policy: str, day: int, inventory_position: float, row: Mapping[str, object]) -> int:
    q = int(round(float(row["Q"])))
    reorder_point = float(row["s"])
    review_period = max(1, int(round(float(row["T"]))))
    order_up_to = int(round(float(row["S"])))

    if policy == "Q,s":
        return q if inventory_position <= reorder_point else 0
    if policy == "T,S":
        return max(0, int(round(order_up_to - inventory_position))) if day % review_period == 0 else 0
    if policy == "s,S":
        return max(0, int(round(order_up_to - inventory_position))) if inventory_position <= reorder_point else 0
    raise ValueError(f"Politica no reconocida: {policy}")


def simulate_policy(
    row: Mapping[str, object],
    policy: str,
    settings: SimulationSettings,
    historical: pd.DataFrame | None = None,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Simula una politica con demanda y lead time estocasticos.

    Se modelan ventas perdidas: la demanda no atendida genera costo de quiebre,
    pero no se arrastra como pedido pendiente al dia siguiente.
    """
    if policy not in POLICY_LABELS:
        raise ValueError(f"Politica no reconocida: {policy}")

    product = str(row["Producto"]).strip()
    rng = np.random.default_rng(_stable_seed(settings.seed, product, policy))
    history = _historical_values(product, historical)
    all_demand = sample_demand(
        row,
        rng,
        (settings.replications, settings.horizon_days),
        history,
    )

    daily_on_hand = np.zeros(settings.horizon_days, dtype=float)
    daily_lost = np.zeros(settings.horizon_days, dtype=float)
    daily_orders = np.zeros(settings.horizon_days, dtype=float)

    total_costs = np.zeros(settings.replications, dtype=float)
    relevant_costs = np.zeros(settings.replications, dtype=float)
    holding_costs = np.zeros(settings.replications, dtype=float)
    ordering_costs = np.zeros(settings.replications, dtype=float)
    shortage_costs = np.zeros(settings.replications, dtype=float)
    purchase_costs = np.zeros(settings.replications, dtype=float)
    fill_rates = np.zeros(settings.replications, dtype=float)
    days_without_stockout = np.zeros(settings.replications, dtype=float)
    average_inventories = np.zeros(settings.replications, dtype=float)
    order_counts = np.zeros(settings.replications, dtype=float)
    stockout_events = np.zeros(settings.replications, dtype=float)
    lost_units = np.zeros(settings.replications, dtype=float)

    unit_cost = float(row["Costo_unitario"])
    order_cost = float(row["Costo_orden"])
    daily_holding_rate = float(row["Tasa_mantenimiento_anual"]) / 365.0
    shortage_cost = float(row["Costo_quiebre_unidad"])

    for rep in range(settings.replications):
        on_hand = int(round(float(row["Stock_inicial"])))
        pipeline: list[tuple[int, int]] = []
        rep_holding = 0.0
        rep_shortage = 0.0
        rep_purchase = 0.0
        rep_ordering = 0.0
        rep_lost = 0
        rep_stockout_days = 0
        rep_orders = 0
        rep_events = 0
        previous_stockout = False
        inventory_sum = 0.0

        for day in range(settings.horizon_days):
            arrivals = sum(qty for arrival_day, qty in pipeline if arrival_day == day)
            if arrivals:
                on_hand += arrivals
            pipeline = [(arrival_day, qty) for arrival_day, qty in pipeline if arrival_day > day]

            demand = int(all_demand[rep, day])
            sold = min(on_hand, demand)
            lost = demand - sold
            on_hand -= sold
            stockout_today = lost > 0
            if stockout_today:
                rep_stockout_days += 1
                rep_lost += lost
                if not previous_stockout:
                    rep_events += 1
            previous_stockout = stockout_today

            on_order = sum(qty for _, qty in pipeline)
            inventory_position = on_hand + on_order
            order_qty = _order_quantity(policy, day, inventory_position, row)
            if order_qty > 0:
                lead_time = sample_lead_time(row, rng)
                pipeline.append((day + lead_time, order_qty))
                rep_orders += 1
                rep_ordering += order_cost
                rep_purchase += order_qty * unit_cost

            holding = on_hand * unit_cost * daily_holding_rate
            stockout = lost * shortage_cost
            rep_holding += holding
            rep_shortage += stockout
            inventory_sum += on_hand

            daily_on_hand[day] += on_hand
            daily_lost[day] += lost
            daily_orders[day] += order_qty

        rep_demand = int(all_demand[rep].sum())
        fill_rates[rep] = 1.0 - (rep_lost / rep_demand) if rep_demand else 1.0
        days_without_stockout[rep] = 1.0 - (rep_stockout_days / settings.horizon_days)
        average_inventories[rep] = inventory_sum / settings.horizon_days
        order_counts[rep] = rep_orders
        stockout_events[rep] = rep_events
        lost_units[rep] = rep_lost
        relevant_costs[rep] = rep_holding + rep_shortage + rep_ordering
        total_costs[rep] = relevant_costs[rep] + rep_purchase
        holding_costs[rep] = rep_holding
        ordering_costs[rep] = rep_ordering
        shortage_costs[rep] = rep_shortage
        purchase_costs[rep] = rep_purchase

    divisor = float(settings.replications)
    trajectory = pd.DataFrame({
        "Dia": np.arange(1, settings.horizon_days + 1),
        "Inventario_promedio": daily_on_hand / divisor,
        "Demanda_no_atendida_promedio": daily_lost / divisor,
        "Cantidad_ordenada_promedio": daily_orders / divisor,
    })
    trajectory.insert(0, "Politica", policy)
    trajectory.insert(0, "Producto", product)

    summary: dict[str, float | str] = {
        "Producto": product,
        "Politica": policy,
        "Nivel_servicio_unidades": float(fill_rates.mean()),
        "Dias_sin_quiebre": float(days_without_stockout.mean()),
        "Inventario_promedio": float(average_inventories.mean()),
        "Unidades_no_atendidas": float(lost_units.mean()),
        "Eventos_quiebre": float(stockout_events.mean()),
        "Ordenes_promedio": float(order_counts.mean()),
        "Costo_mantenimiento_promedio": float(holding_costs.mean()),
        "Costo_ordenamiento_promedio": float(ordering_costs.mean()),
        "Costo_quiebre_promedio": float(shortage_costs.mean()),
        "Costo_compras_promedio": float(purchase_costs.mean()),
        "Costo_relevante_promedio": float(relevant_costs.mean()),
        "Costo_total_promedio": float(total_costs.mean()),
        "Costo_total_p95": float(np.quantile(total_costs, 0.95)),
        "Probabilidad_servicio_objetivo": float(np.mean(fill_rates >= settings.target_probability)),
    }
    return summary, trajectory


def run_experiment(
    products: pd.DataFrame,
    policies: Iterable[str],
    settings: SimulationSettings,
    historical: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    stock_summaries: list[dict[str, float | str]] = []
    percentile_tables: list[pd.DataFrame] = []
    policy_summaries: list[dict[str, float | str]] = []
    trajectories: list[pd.DataFrame] = []

    for _, row in products.iterrows():
        stock_summary, percentiles = validate_determined_stock(row, settings, historical)
        stock_summaries.append(stock_summary)
        percentile_tables.append(percentiles)
        for policy in policies:
            policy_summary, trajectory = simulate_policy(row, policy, settings, historical)
            policy_summaries.append(policy_summary)
            trajectories.append(trajectory)

    stock_df = pd.DataFrame(stock_summaries)
    policies_df = pd.DataFrame(policy_summaries)
    if not policies_df.empty:
        policies_df["Cumple_objetivo"] = policies_df["Nivel_servicio_unidades"] >= settings.target_probability
        policies_df["Ranking_costo_relevante"] = policies_df.groupby("Producto")["Costo_relevante_promedio"].rank(method="dense")
        policies_df["Recomendada"] = False
        for product, group in policies_df.groupby("Producto"):
            eligible = group[group["Cumple_objetivo"]]
            pool = eligible if not eligible.empty else group.sort_values(
                ["Nivel_servicio_unidades", "Costo_relevante_promedio"], ascending=[False, True]
            ).head(1)
            idx = pool["Costo_relevante_promedio"].idxmin()
            policies_df.loc[idx, "Recomendada"] = True

    return {
        "validacion_stock": stock_df,
        "percentiles_demanda": pd.concat(percentile_tables, ignore_index=True) if percentile_tables else pd.DataFrame(),
        "comparacion_politicas": policies_df,
        "trayectorias": pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame(),
    }
