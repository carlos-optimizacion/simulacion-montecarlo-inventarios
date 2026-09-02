from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO
import time

import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    POLICY_LABELS,
    REQUIRED_PRODUCT_COLUMNS,
    SimulationSettings,
    run_experiment,
    validate_products,
)


st.set_page_config(
    page_title="Monte Carlo | Gestión de Inventarios",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --slate: #43546d;
        --slate-dark: #2e3b4f;
        --ink: #172033;
        --blue: #4d78a8;
    }
    .stApp { background: #f5f7fa; color: var(--ink); }
    .block-container { max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem; }
    .mc-hero {
        background: linear-gradient(125deg, var(--slate-dark), var(--slate) 62%, #62738b);
        border: 1px solid #718096;
        border-radius: 8px;
        padding: 1.45rem 1.8rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 22px rgba(35, 48, 68, .16);
    }
    .mc-hero h1 { color: white; font-size: 1.95rem; font-weight: 500; margin: 0; letter-spacing: .035em; }
    .mc-hero p { color: #e8eef6; margin: .4rem 0 0; font-size: .98rem; }
    .mc-ribbon {
        background: var(--slate);
        color: white;
        border-radius: 6px;
        padding: .72rem 1rem;
        margin: .35rem 0 1rem;
        font-size: 1.05rem;
        letter-spacing: .02em;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbe2ea;
        border-top: 4px solid var(--slate);
        border-radius: 7px;
        padding: .8rem .95rem;
        min-height: 124px;
        box-shadow: 0 4px 14px rgba(38, 51, 69, .07);
    }
    div[data-testid="stMetricLabel"] { color: #5b6677; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--ink); }
    div[data-testid="stDataFrame"] { border: 1px solid #d7dee8; border-radius: 6px; }
    .mc-note {
        background: #eef3f8;
        border-left: 4px solid var(--blue);
        border-radius: 4px;
        padding: .8rem 1rem;
        margin: .35rem 0 1rem;
    }
    [data-testid="stSidebar"] { background: #eef2f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_PRODUCTS = pd.DataFrame([
    {
        "Producto": "Producto A", "Stock_inicial": 210,
        "Distribucion_demanda": "Normal", "Demanda_media": 7,
        "Demanda_desv": 2, "Demanda_min": 2, "Demanda_moda": 6,
        "Demanda_max": 12, "Lead_time_media": 5, "Lead_time_desv": 1,
        "Costo_unitario": 20, "Costo_orden": 45,
        "Tasa_mantenimiento_anual": 0.24, "Costo_quiebre_unidad": 18,
        "Q": 80, "s": 48, "T": 7, "S": 110,
    },
    {
        "Producto": "Producto B", "Stock_inicial": 140,
        "Distribucion_demanda": "Poisson", "Demanda_media": 4,
        "Demanda_desv": 0, "Demanda_min": 1, "Demanda_moda": 4,
        "Demanda_max": 9, "Lead_time_media": 4, "Lead_time_desv": 1,
        "Costo_unitario": 35, "Costo_orden": 50,
        "Tasa_mantenimiento_anual": 0.22, "Costo_quiebre_unidad": 25,
        "Q": 50, "s": 28, "T": 7, "S": 75,
    },
])


def initialize_state() -> None:
    if "products_config" not in st.session_state:
        st.session_state["products_config"] = DEFAULT_PRODUCTS.copy()
    if "historical_config" not in st.session_state:
        st.session_state["historical_config"] = None


def make_template(products: pd.DataFrame | None = None) -> bytes:
    products = DEFAULT_PRODUCTS if products is None else products
    rng = np.random.default_rng(2026)
    historical_rows = []
    for product, mean in [("Producto A", 7), ("Producto B", 4)]:
        for day in pd.date_range("2026-01-01", periods=60, freq="D"):
            historical_rows.append({
                "Fecha": day, "Producto": product,
                "Demanda": max(0, int(rng.poisson(mean))),
            })
    descriptions = [
        "Nombre único del producto", "Stock que se desea evaluar al iniciar",
        "Normal, Poisson, Triangular o Empírica", "Promedio diario de demanda",
        "Desviación estándar diaria (Normal)", "Valor mínimo (Triangular)",
        "Valor más probable o moda (Triangular)", "Valor máximo (Triangular)",
        "Plazo promedio de reposición en días", "Desviación del plazo de reposición",
        "Costo de compra por unidad", "Costo administrativo por orden",
        "Porcentaje anual en decimal; ejemplo 0.24",
        "Penalidad o margen perdido por unidad no atendida",
        "Cantidad fija de pedido para (Q,s)", "Punto de pedido para (Q,s) y (s,S)",
        "Intervalo de revisión en días para (T,S)", "Nivel máximo para (T,S) y (s,S)",
    ]
    dictionary = pd.DataFrame({"Campo": REQUIRED_PRODUCT_COLUMNS, "Descripción": descriptions})
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        products.to_excel(writer, sheet_name="Productos", index=False)
        pd.DataFrame(historical_rows).to_excel(writer, sheet_name="Demanda_historica", index=False)
        dictionary.to_excel(writer, sheet_name="Diccionario", index=False)
    return buffer.getvalue()


def read_workbook(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    workbook = pd.ExcelFile(uploaded_file)
    if "Productos" not in workbook.sheet_names:
        raise ValueError("El archivo debe contener una hoja llamada Productos.")
    products = pd.read_excel(workbook, sheet_name="Productos")
    historical = (
        pd.read_excel(workbook, sheet_name="Demanda_historica")
        if "Demanda_historica" in workbook.sheet_names else None
    )
    return products, historical


def duration_to_days(duration: int, unit: str) -> int:
    return int(duration * {"Días": 1, "Semanas": 7, "Meses": 30}[unit])


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_currency(value: float) -> str:
    return f"S/ {value:,.2f}"


def results_to_excel(
    results: dict[str, pd.DataFrame],
    settings: SimulationSettings,
    period: dict[str, object],
) -> bytes:
    buffer = BytesIO()
    parameters = pd.DataFrame({
        "Parámetro": [
            "Fecha_inicio", "Fecha_fin", "Duracion_ingresada", "Unidad_tiempo",
            "Horizonte_dias", "Frecuencia_visualizacion", "Iteraciones",
            "Periodo_proteccion_dias", "Nivel_objetivo", "Semilla", "Tiempo_ejecucion_segundos",
        ],
        "Valor": [
            period["start_date"], period["end_date"], period["duration"], period["unit"],
            settings.horizon_days, period["frequency"], settings.replications,
            settings.protection_days, settings.target_probability, settings.seed,
            period["run_seconds"],
        ],
    })
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        parameters.to_excel(writer, sheet_name="Parametros", index=False)
        for name, frame in results.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def histogram_frame(values: pd.Series) -> pd.DataFrame:
    bins = min(24, max(8, int(np.sqrt(len(values)))))
    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    labels = [f"{edges[i]:.0f}–{edges[i + 1]:.0f}" for i in range(len(counts))]
    return pd.DataFrame({"Rango de demanda": labels, "Escenarios": counts})


def inventory_chart(trajectories: pd.DataFrame, start_date: date, frequency: str) -> pd.DataFrame:
    frame = trajectories.copy()
    frame["Fecha"] = pd.Timestamp(start_date) + pd.to_timedelta(frame["Dia"] - 1, unit="D")
    if frequency == "Diaria":
        frame["Periodo"] = frame["Fecha"].dt.strftime("%d/%m/%Y")
    elif frequency == "Semanal":
        frame["Periodo_n"] = ((frame["Dia"] - 1) // 7) + 1
        frame["Periodo"] = "Semana " + frame["Periodo_n"].astype(str)
    else:
        frame["Periodo_n"] = ((frame["Dia"] - 1) // 30) + 1
        frame["Periodo"] = "Mes " + frame["Periodo_n"].astype(str)
    grouped = frame.groupby(["Periodo", "Politica"], sort=False)["Inventario_promedio"].mean().reset_index()
    return grouped.pivot(index="Periodo", columns="Politica", values="Inventario_promedio")


def result_context(product_key: str):
    results = st.session_state["results"]
    settings = st.session_state["settings"]
    period = st.session_state["period_config"]
    stock = results["validacion_stock"]
    selected_product = st.selectbox("Producto analizado", stock["Producto"].tolist(), key=product_key)
    selected_stock = stock[stock["Producto"] == selected_product].iloc[0]
    comparison = results["comparacion_politicas"]
    product_comparison = comparison[comparison["Producto"] == selected_product].copy()
    recommended = product_comparison[product_comparison["Recomendada"]]
    best = recommended.iloc[0] if not recommended.empty else product_comparison.iloc[0]
    return results, settings, period, selected_product, selected_stock, product_comparison, best


def require_results() -> bool:
    if "results" in st.session_state:
        return True
    st.warning("Primero ejecute el experimento en la página **Simulación Monte Carlo**.")
    return False


def render_data_page() -> None:
    st.markdown('<div class="mc-ribbon">1. DATOS Y CONFIGURACIÓN DEL INVENTARIO</div>', unsafe_allow_html=True)
    st.caption("Defina productos, demanda, costos, lead time y parámetros de las políticas de inventario.")
    left, right = st.columns([2, 1])
    with left:
        source = st.radio(
            "Modalidad de ingreso", ["Ingreso manual", "Cargar archivo Excel"],
            horizontal=True, key="data_source",
        )
    with right:
        st.download_button(
            "⬇️ Descargar plantilla Excel",
            data=make_template(st.session_state["products_config"]),
            file_name="plantilla_montecarlo_inventarios.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    initial_products = st.session_state["products_config"].copy()
    historical_data = st.session_state["historical_config"]
    editor_key = "products_editor_manual"
    if source == "Cargar archivo Excel":
        upload = st.file_uploader("Seleccione la plantilla completada", type=["xlsx"])
        editor_key = "products_editor_excel"
        if upload is not None:
            try:
                initial_products, historical_data = read_workbook(upload)
                editor_key = f"products_editor_excel_{upload.name}_{upload.size}"
                st.success(f"Archivo cargado: {len(initial_products)} producto(s).")
            except Exception as exc:
                st.error(str(exc))

    column_config = {
        "Distribucion_demanda": st.column_config.SelectboxColumn(
            "Distribución", options=["Normal", "Poisson", "Triangular", "Empirica"]
        ),
        "Tasa_mantenimiento_anual": st.column_config.NumberColumn(
            "Tasa mantenimiento anual", help="Ejemplo: 0.24 representa 24% anual.",
            min_value=0.0, format="%.4f",
        ),
    }
    products = st.data_editor(
        initial_products, num_rows="dynamic", use_container_width=True,
        hide_index=True, column_config=column_config, key=editor_key,
    )

    if source == "Ingreso manual":
        with st.expander("Demanda histórica opcional para distribución empírica"):
            history_base = historical_data if historical_data is not None else pd.DataFrame(columns=["Fecha", "Producto", "Demanda"])
            historical_editor = st.data_editor(
                history_base, num_rows="dynamic", use_container_width=True, hide_index=True,
                column_config={
                    "Fecha": st.column_config.DateColumn("Fecha"),
                    "Demanda": st.column_config.NumberColumn("Demanda", min_value=0, step=1),
                },
                key="historical_manual_editor",
            )
            historical_data = (
                historical_editor.dropna(subset=["Producto", "Demanda"]).copy()
                if not historical_editor.empty else None
            )

    if st.button("💾 Guardar configuración", type="primary", use_container_width=True):
        errors = validate_products(products, historical_data)
        if errors:
            for error in errors:
                st.error(error)
        else:
            st.session_state["products_config"] = products.copy()
            st.session_state["historical_config"] = historical_data
            st.session_state.pop("results", None)
            st.session_state.pop("settings", None)
            st.session_state.pop("period_config", None)
            st.success(f"Configuración guardada para {len(products)} producto(s). Ya puede ejecutar la simulación.")

    with st.expander("Guía de las políticas"):
        st.markdown(
            "- **(Q,s):** al llegar a `s`, solicita una cantidad fija `Q`.\n"
            "- **(T,S):** cada `T` días revisa y repone hasta `S`.\n"
            "- **(s,S):** al llegar a `s`, repone una cantidad variable hasta `S`."
        )


def render_simulation_page() -> None:
    st.markdown('<div class="mc-ribbon">2. SIMULACIÓN MONTE CARLO</div>', unsafe_allow_html=True)
    st.caption("Configure el tiempo que desea representar y el número de futuros posibles que generará el modelo.")

    st.markdown("#### Tiempo del experimento")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        start_date = st.date_input("Fecha de inicio", value=date.today(), key="sim_start_date")
    with t2:
        unit = st.selectbox(
            "Unidad de tiempo", ["Días", "Semanas", "Meses"], index=2, key="sim_unit"
        )
    with t3:
        duration_limits = {"Días": (730, 180), "Semanas": (104, 26), "Meses": (24, 6)}
        maximum_duration, default_duration = duration_limits[unit]
        duration = st.number_input(
            "Duración", 1, maximum_duration, default_duration, 1,
            key=f"sim_duration_{unit}",
        )
    with t4:
        frequency = st.selectbox("Visualización", ["Diaria", "Semanal", "Mensual"], key="sim_frequency")

    horizon_days = duration_to_days(int(duration), unit)
    end_date = start_date + timedelta(days=horizon_days - 1)
    st.markdown(
        f"<div class='mc-note'><b>Horizonte representado:</b> {duration} {unit.lower()} = "
        f"<b>{horizon_days:,} días</b>, desde {start_date.strftime('%d/%m/%Y')} hasta "
        f"{end_date.strftime('%d/%m/%Y')}.</div>", unsafe_allow_html=True,
    )

    st.markdown("#### Escenarios y nivel de protección")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        replications = st.number_input("Iteraciones Monte Carlo", 100, 5000, 1000, 100, key="sim_replications")
    with c2:
        protection = st.number_input("Periodo de protección (días)", 1, 365, 30, 1, key="sim_protection")
    with c3:
        target = st.slider("Nivel de servicio objetivo", 0.80, 0.999, 0.95, 0.001, key="sim_target")
    with c4:
        seed = st.number_input("Semilla reproducible", 0, 999999, 2026, 1, key="sim_seed")

    policies = st.multiselect(
        "Políticas que desea simular", options=list(POLICY_LABELS),
        default=list(POLICY_LABELS), format_func=lambda key: POLICY_LABELS[key],
        key="sim_policies",
    )
    products = st.session_state["products_config"]
    historical = st.session_state["historical_config"]
    st.caption(f"Configuración activa: {len(products)} producto(s). El horizonte y las iteraciones son conceptos independientes.")

    if st.button("▶️ Ejecutar simulación Monte Carlo", type="primary", use_container_width=True):
        errors = validate_products(products, historical)
        if not policies:
            errors.append("Seleccione al menos una política de inventario.")
        if protection > horizon_days:
            errors.append("El periodo de protección no puede superar el horizonte de simulación.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            settings = SimulationSettings(
                horizon_days=horizon_days, replications=int(replications),
                protection_days=int(protection), target_probability=float(target), seed=int(seed),
            )
            started = time.perf_counter()
            with st.spinner("Generando escenarios de demanda, inventario y costos..."):
                results = run_experiment(products, policies, settings, historical)
            elapsed = time.perf_counter() - started
            st.session_state["results"] = results
            st.session_state["settings"] = settings
            st.session_state["period_config"] = {
                "start_date": start_date, "end_date": end_date,
                "duration": int(duration), "unit": unit,
                "frequency": frequency, "run_seconds": elapsed,
            }
            st.success(f"Simulación completada en {elapsed:.2f} segundos.")

    if not require_results():
        return

    results, settings, period, selected_product, selected_stock, product_comparison, best = result_context("sim_product")
    st.markdown(
        f"<div class='mc-note'><b>Experimento vigente:</b> {settings.replications:,} escenarios · "
        f"{settings.horizon_days:,} días simulados · {settings.protection_days} días de protección · "
        f"cálculo real {period['run_seconds']:.2f} s.</div>", unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Escenarios generados", f"{settings.replications:,}")
    m2.metric("Horizonte simulado", f"{settings.horizon_days:,} días")
    m3.metric("Cobertura del stock", format_percent(selected_stock["Probabilidad_cobertura"]))
    m4.metric("Probabilidad de quiebre", format_percent(selected_stock["Probabilidad_quiebre"]))
    m5.metric("Tiempo de cálculo", f"{period['run_seconds']:.2f} s")

    scenarios = results["escenarios_demanda"]
    scenarios = scenarios[scenarios["Producto"] == selected_product].copy()
    percentiles = results["percentiles_demanda"]
    percentiles = percentiles[percentiles["Producto"] == selected_product].set_index("Percentil")

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### Distribución de demanda en {settings.protection_days} días")
        hist = histogram_frame(scenarios["Demanda_acumulada"])
        st.bar_chart(hist, x="Rango de demanda", y="Escenarios", color="#4d78a8", use_container_width=True)
        st.caption("Cada barra agrupa futuros posibles. La dispersión muestra la variabilidad que debe soportar el stock.")
    with right:
        st.markdown("#### Demanda acumulada por percentil")
        st.bar_chart(percentiles["Demanda_acumulada"], color="#62738b", use_container_width=True)
        st.caption("El percentil objetivo determina el stock requerido durante el periodo de protección.")

    st.markdown(f"#### Evolución del inventario durante {period['duration']} {str(period['unit']).lower()}")
    trajectories = results["trayectorias"]
    trajectories = trajectories[trajectories["Producto"] == selected_product]
    chart = inventory_chart(trajectories, period["start_date"], str(period["frequency"]))
    st.line_chart(chart, use_container_width=True)

    st.markdown("#### Evidencia de las iteraciones Monte Carlo")
    rows = st.selectbox("Iteraciones visibles", [25, 50, 100, 250], index=1, key="scenario_rows")
    scenario_view = scenarios.head(rows).copy()
    scenario_view["Cobertura"] = scenario_view["Cobertura"].map({True: "Sí", False: "No"})
    scenario_view["Dia_quiebre"] = scenario_view["Dia_quiebre"].apply(
        lambda value: "Sin quiebre" if pd.isna(value) else int(value)
    )
    st.dataframe(scenario_view, use_container_width=True, hide_index=True)


def render_policies_page() -> None:
    st.markdown('<div class="mc-ribbon">3. POLÍTICAS, SERVICIO Y COSTOS</div>', unsafe_allow_html=True)
    if not require_results():
        return
    results, settings, period, selected_product, selected_stock, comparison, best = result_context("policy_product")
    st.info(
        f"La política sugerida para **{selected_product}** es **{POLICY_LABELS[best['Politica']]}**. "
        f"Su nivel de servicio esperado es {best['Nivel_servicio_unidades']:.1%}."
    )

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Nivel de servicio", format_percent(best["Nivel_servicio_unidades"]), f"{best['Brecha_nivel_servicio']:+.1%} vs objetivo")
    p2.metric("Quiebre en el horizonte", format_percent(best["Probabilidad_quiebre_horizonte"]))
    p3.metric("Inventario promedio", f"{best['Inventario_promedio']:.1f} u.")
    p4.metric("Costo relevante", format_currency(best["Costo_relevante_promedio"]))

    left, right = st.columns(2)
    with left:
        st.markdown("#### Frontera costo–servicio")
        st.scatter_chart(
            comparison, x="Costo_relevante_promedio", y="Nivel_servicio_unidades",
            color="Politica", size="Inventario_promedio", use_container_width=True,
        )
    with right:
        st.markdown("#### Composición del costo")
        breakdown = comparison.set_index("Politica")[[
            "Costo_mantenimiento_promedio", "Costo_ordenamiento_promedio", "Costo_quiebre_promedio"
        ]]
        breakdown.columns = ["Mantenimiento", "Ordenamiento", "Quiebre"]
        st.bar_chart(breakdown, use_container_width=True)

    policy_view = comparison[[
        "Politica", "Nivel_servicio_unidades", "Probabilidad_quiebre_horizonte",
        "Inventario_promedio", "Ordenes_promedio", "Rotacion_anualizada",
        "Dias_cobertura_promedio", "Costo_por_unidad_atendida",
        "Costo_relevante_promedio", "Costo_relevante_CVaR95", "Recomendada",
    ]].copy()
    policy_view["Politica"] = policy_view["Politica"].map(POLICY_LABELS)
    st.dataframe(
        policy_view.style.format({
            "Nivel_servicio_unidades": "{:.2%}", "Probabilidad_quiebre_horizonte": "{:.2%}",
            "Inventario_promedio": "{:.1f}", "Ordenes_promedio": "{:.1f}",
            "Rotacion_anualizada": "{:.1f}", "Dias_cobertura_promedio": "{:.1f}",
            "Costo_por_unidad_atendida": "S/ {:,.2f}",
            "Costo_relevante_promedio": "S/ {:,.2f}",
            "Costo_relevante_CVaR95": "S/ {:,.2f}",
        }), use_container_width=True, hide_index=True,
    )


def render_dashboard_page() -> None:
    st.markdown('<div class="mc-ribbon">4. DASHBOARD EJECUTIVO</div>', unsafe_allow_html=True)
    if not require_results():
        return
    results, settings, period, selected_product, selected_stock, comparison, best = result_context("dashboard_product")
    st.markdown(
        f"<div class='mc-note'><b>Alcance:</b> {period['duration']} {str(period['unit']).lower()} · "
        f"{settings.replications:,} escenarios · objetivo {settings.target_probability:.1%} · "
        f"desde {period['start_date'].strftime('%d/%m/%Y')} hasta {period['end_date'].strftime('%d/%m/%Y')}.</div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Cobertura del stock definido", format_percent(selected_stock["Probabilidad_cobertura"]),
        f"{selected_stock['Probabilidad_cobertura'] - settings.target_probability:+.1%} vs objetivo",
    )
    k2.metric(
        "Stock requerido", f"{selected_stock['Stock_recomendado_objetivo']:.0f} u.",
        f"{selected_stock['Brecha_stock']:+.0f} u. vs requerido",
    )
    k3.metric(
        f"Servicio con {best['Politica']}", format_percent(best["Nivel_servicio_unidades"]),
        f"{best['Brecha_nivel_servicio']:+.1%} vs objetivo",
    )
    k4.metric(
        "Costo relevante esperado", format_currency(best["Costo_relevante_promedio"]),
        f"CVaR 95% {format_currency(best['Costo_relevante_CVaR95'])}", delta_color="off",
    )

    criterion = (
        "cumple el objetivo y minimiza el costo relevante"
        if bool(best["Cumple_objetivo"]) else "alcanza el mayor nivel de servicio disponible"
    )
    st.info(
        f"**Decisión sugerida:** aplicar **{POLICY_LABELS[best['Politica']]}**, porque {criterion}. "
        f"El stock inicial es **{selected_stock['Evaluacion'].lower()}** para el periodo de protección."
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Stock de seguridad", f"{selected_stock['Stock_seguridad_recomendado']:.0f} u.")
    d2.metric("Demanda CVaR 95%", f"{selected_stock['Demanda_CVaR95_periodo']:.0f} u.")
    d3.metric("Rotación anualizada", f"{best['Rotacion_anualizada']:.1f} veces")
    d4.metric("Cobertura promedio", f"{best['Dias_cobertura_promedio']:.1f} días")

    definitions = pd.DataFrame([
        ("Cobertura del stock", "Probabilidad de cubrir toda la demanda del periodo de protección."),
        ("Fill rate", "Unidades atendidas inmediatamente entre unidades demandadas."),
        ("VaR 95%", "Umbral que no se supera en aproximadamente 95% de los escenarios."),
        ("CVaR 95%", "Promedio del 5% de escenarios más adversos."),
        ("Costo relevante", "Mantenimiento + ordenamiento + quiebre."),
    ], columns=["Indicador", "Definición operativa"])
    with st.expander("Definiciones y trazabilidad"):
        st.dataframe(definitions, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar resultados e indicadores avanzados en Excel",
        data=results_to_excel(results, settings, period),
        file_name="resultados_avanzados_montecarlo_inventarios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


initialize_state()

st.markdown(
    """
    <div class="mc-hero">
      <h1>SIMULACIÓN MONTE CARLO · GESTIÓN DE INVENTARIOS</h1>
      <p>Demanda estocástica, validación del stock, políticas de reposición, riesgo y costo.</p>
    </div>
    """, unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Navegación")
    page = st.radio(
        "Seleccione una hoja",
        [
            "1. Datos y configuración", "2. Simulación Monte Carlo",
            "3. Políticas y costos", "4. Dashboard ejecutivo",
        ], label_visibility="collapsed",
    )
    st.divider()
    st.caption("La simulación tiene una hoja independiente para mostrar escenarios, horizonte e iteraciones.")

if page == "1. Datos y configuración":
    render_data_page()
elif page == "2. Simulación Monte Carlo":
    render_simulation_page()
elif page == "3. Políticas y costos":
    render_policies_page()
else:
    render_dashboard_page()

st.divider()
st.caption(
    "Herramienta de apoyo para decisiones. La calidad de los resultados depende de que la demanda, "
    "los costos y el lead time representen adecuadamente la operación."
)
