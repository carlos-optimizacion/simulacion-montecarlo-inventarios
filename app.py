from __future__ import annotations

from io import BytesIO

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
    page_title="Monte Carlo | Gestion de Inventarios",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --slate: #43546d;
        --slate-dark: #2e3b4f;
        --ice: #e9eef5;
        --ink: #172033;
        --blue: #4d78a8;
        --green: #2f855a;
        --amber: #c58a18;
        --red: #c65353;
    }
    .stApp { background: #f5f7fa; color: var(--ink); }
    .block-container { max-width: 1500px; padding-top: 1.25rem; padding-bottom: 3rem; }
    .mc-hero {
        background: linear-gradient(125deg, var(--slate-dark), var(--slate) 62%, #62738b);
        border: 1px solid #718096;
        border-radius: 8px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 22px rgba(35, 48, 68, .16);
    }
    .mc-hero h1 { color: white; font-size: 2.05rem; font-weight: 500; margin: 0; letter-spacing: .035em; }
    .mc-hero p { color: #e8eef6; margin: .45rem 0 0; font-size: 1rem; }
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
        padding: .85rem 1rem;
        min-height: 128px;
        box-shadow: 0 4px 14px rgba(38, 51, 69, .07);
    }
    div[data-testid="stMetricLabel"] { color: #5b6677; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: var(--ink); }
    div[data-testid="stTabs"] button { font-weight: 600; }
    div[data-testid="stDataFrame"] { border: 1px solid #d7dee8; border-radius: 6px; }
    .mc-note {
        background: #eef3f8;
        border-left: 4px solid var(--blue);
        border-radius: 4px;
        padding: .8rem 1rem;
        margin: .4rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_PRODUCTS = pd.DataFrame([
    {
        "Producto": "Producto A",
        "Stock_inicial": 210,
        "Distribucion_demanda": "Normal",
        "Demanda_media": 7,
        "Demanda_desv": 2,
        "Demanda_min": 2,
        "Demanda_moda": 6,
        "Demanda_max": 12,
        "Lead_time_media": 5,
        "Lead_time_desv": 1,
        "Costo_unitario": 20,
        "Costo_orden": 45,
        "Tasa_mantenimiento_anual": 0.24,
        "Costo_quiebre_unidad": 18,
        "Q": 80,
        "s": 48,
        "T": 7,
        "S": 110,
    },
    {
        "Producto": "Producto B",
        "Stock_inicial": 140,
        "Distribucion_demanda": "Poisson",
        "Demanda_media": 4,
        "Demanda_desv": 0,
        "Demanda_min": 1,
        "Demanda_moda": 4,
        "Demanda_max": 9,
        "Lead_time_media": 4,
        "Lead_time_desv": 1,
        "Costo_unitario": 35,
        "Costo_orden": 50,
        "Tasa_mantenimiento_anual": 0.22,
        "Costo_quiebre_unidad": 25,
        "Q": 50,
        "s": 28,
        "T": 7,
        "S": 75,
    },
])


def make_template(products: pd.DataFrame | None = None) -> bytes:
    products = DEFAULT_PRODUCTS if products is None else products
    rng = np.random.default_rng(2026)
    historical_rows = []
    for product, mean in [("Producto A", 7), ("Producto B", 4)]:
        for day in pd.date_range("2026-01-01", periods=60, freq="D"):
            historical_rows.append({
                "Fecha": day,
                "Producto": product,
                "Demanda": max(0, int(rng.poisson(mean))),
            })
    instructions = pd.DataFrame({
        "Campo": REQUIRED_PRODUCT_COLUMNS,
        "Descripcion": [
            "Nombre unico del producto",
            "Stock que se desea evaluar al iniciar",
            "Normal, Poisson, Triangular o Empirica",
            "Promedio diario de demanda",
            "Desviacion estandar diaria (Normal)",
            "Valor minimo (Triangular)",
            "Valor mas probable o moda (Triangular)",
            "Valor maximo (Triangular)",
            "Plazo promedio de reposicion en dias",
            "Desviacion del plazo de reposicion",
            "Costo de compra por unidad",
            "Costo administrativo por orden",
            "Porcentaje anual en decimal; ejemplo 0.24",
            "Penalidad o margen perdido por unidad no atendida",
            "Cantidad fija de pedido para (Q,s)",
            "Punto de pedido para (Q,s) y (s,S)",
            "Intervalo de revision en dias para (T,S)",
            "Nivel maximo para (T,S) y (s,S)",
        ],
    })
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        products.to_excel(writer, sheet_name="Productos", index=False)
        pd.DataFrame(historical_rows).to_excel(writer, sheet_name="Demanda_historica", index=False)
        instructions.to_excel(writer, sheet_name="Diccionario", index=False)
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


def results_to_excel(results: dict[str, pd.DataFrame], settings: SimulationSettings) -> bytes:
    buffer = BytesIO()
    parameters = pd.DataFrame({
        "Parametro": ["Horizonte_dias", "Iteraciones", "Periodo_proteccion_dias", "Nivel_objetivo", "Semilla"],
        "Valor": [settings.horizon_days, settings.replications, settings.protection_days, settings.target_probability, settings.seed],
    })
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        parameters.to_excel(writer, sheet_name="Parametros", index=False)
        for name, frame in results.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    return buffer.getvalue()


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_currency(value: float) -> str:
    return f"S/ {value:,.2f}"


st.markdown(
    """
    <div class="mc-hero">
      <h1>SIMULACIÓN MONTE CARLO · GESTIÓN DE INVENTARIOS</h1>
      <p>Evaluación estocástica de stock, nivel de servicio, riesgo y costo de las políticas (Q,s), (T,S) y (s,S).</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Como interpreta el modelo", expanded=False):
    st.markdown(
        """
        1. **Validacion del stock:** simula la demanda acumulada durante el periodo de proteccion, sin reposicion, y calcula la probabilidad de que el stock sea suficiente.
        2. **Politicas:** simula reposiciones durante el horizonte completo con lead time aleatorio.
        3. **Nivel de servicio:** porcentaje de unidades demandadas que se atienden inmediatamente (*fill rate*).
        4. **Costo relevante:** mantenimiento + ordenamiento + quiebre. Se usa para comparar politicas porque aisla los costos controlables.
        5. **Costo total:** costo relevante + compras realizadas durante el horizonte.

        El modelo considera **ventas perdidas**: una unidad no atendida genera costo de quiebre y no queda pendiente para otro dia.
        """
    )

st.subheader("1. Datos de entrada")
left, right = st.columns([2, 1])
with left:
    source = st.radio(
        "Modalidad de ingreso",
        ["Ingreso manual", "Cargar archivo Excel"],
        horizontal=True,
    )
with right:
    st.download_button(
        "⬇️ Descargar plantilla Excel",
        data=make_template(),
        file_name="plantilla_montecarlo_inventarios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

historical_data: pd.DataFrame | None = None
editor_key = f"products_editor_{source}"
if source == "Cargar archivo Excel":
    upload = st.file_uploader("Seleccione la plantilla completada", type=["xlsx"])
    if upload is not None:
        try:
            initial_products, historical_data = read_workbook(upload)
            editor_key = f"products_editor_excel_{upload.name}_{upload.size}"
            st.success(f"Archivo cargado: {len(initial_products)} producto(s).")
        except Exception as exc:
            st.error(str(exc))
            initial_products = DEFAULT_PRODUCTS.copy()
    else:
        initial_products = DEFAULT_PRODUCTS.copy()
        st.info("Mientras carga un archivo, se muestran datos demostrativos editables.")
else:
    initial_products = DEFAULT_PRODUCTS.copy()

column_config = {
    "Distribucion_demanda": st.column_config.SelectboxColumn(
        "Distribucion_demanda",
        options=["Normal", "Poisson", "Triangular", "Empirica"],
    ),
    "Tasa_mantenimiento_anual": st.column_config.NumberColumn(
        "Tasa_mantenimiento_anual",
        help="Ejemplo: ingrese 0.24 para representar 24% anual.",
        min_value=0.0,
        format="%.4f",
    ),
}
products = st.data_editor(
    initial_products,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
    key=editor_key,
)

if source == "Ingreso manual":
    with st.expander("Demanda historica opcional para distribucion Empirica"):
        st.caption(
            "Agregue una fila por observacion diaria. Solo es necesaria cuando algun "
            "producto utiliza la distribucion Empirica."
        )
        historical_editor = st.data_editor(
            pd.DataFrame(columns=["Fecha", "Producto", "Demanda"]),
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Fecha": st.column_config.DateColumn("Fecha"),
                "Demanda": st.column_config.NumberColumn("Demanda", min_value=0, step=1),
            },
            key="historical_manual_editor",
        )
        if not historical_editor.empty and historical_editor[["Producto", "Demanda"]].notna().any(axis=None):
            historical_data = historical_editor.dropna(subset=["Producto", "Demanda"]).copy()

with st.expander("Guia rapida de parametros de las politicas"):
    st.markdown(
        """
        - **(Q,s):** cuando la posicion de inventario es menor o igual que `s`, se solicita una cantidad fija `Q`.
        - **(T,S):** cada `T` dias se revisa el inventario y se ordena lo necesario para alcanzar `S`.
        - **(s,S):** cuando la posicion de inventario llega a `s`, se ordena una cantidad variable hasta alcanzar `S`.
        """
    )

st.subheader("2. Configuracion del experimento")
c1, c2, c3, c4 = st.columns(4)
with c1:
    horizon = st.number_input("Horizonte de simulacion (dias)", 30, 730, 180, 10)
with c2:
    replications = st.number_input("Iteraciones Monte Carlo", 100, 5000, 1000, 100)
with c3:
    protection = st.number_input("Periodo para validar stock (dias)", 1, 365, 30, 1)
with c4:
    target = st.slider("Nivel de servicio objetivo", 0.80, 0.999, 0.95, 0.001)

policies = st.multiselect(
    "Politicas que desea comparar",
    options=list(POLICY_LABELS),
    default=list(POLICY_LABELS),
    format_func=lambda key: POLICY_LABELS[key],
)
seed = st.number_input("Semilla aleatoria para reproducibilidad", 0, 999999, 2026, 1)

run = st.button("▶️ Ejecutar simulacion", type="primary", use_container_width=True)
if run:
    errors = validate_products(products, historical_data)
    if not policies:
        errors.append("Seleccione al menos una politica de inventario.")
    if protection > horizon:
        errors.append("El periodo de proteccion no puede superar el horizonte de simulacion.")
    if errors:
        for error in errors:
            st.error(error)
    else:
        settings = SimulationSettings(
            horizon_days=int(horizon),
            replications=int(replications),
            protection_days=int(protection),
            target_probability=float(target),
            seed=int(seed),
        )
        with st.spinner("Ejecutando escenarios estocasticos..."):
            results = run_experiment(products, policies, settings, historical_data)
        st.session_state["results"] = results
        st.session_state["settings"] = settings
        st.success("Simulacion completada.")

if "results" in st.session_state:
    results = st.session_state["results"]
    settings = st.session_state["settings"]
    stock = results["validacion_stock"]
    comparison = results["comparacion_politicas"]
    trajectories = results["trayectorias"]

    st.divider()
    st.markdown('<div class="mc-ribbon">TABLERO EJECUTIVO DE LA SIMULACIÓN</div>', unsafe_allow_html=True)
    selector_col, scope_col = st.columns([2, 3])
    with selector_col:
        selected_product = st.selectbox("Producto analizado", stock["Producto"].tolist())
    with scope_col:
        st.markdown(
            f"<div class='mc-note'><b>Escenario:</b> {settings.replications:,} iteraciones · "
            f"{settings.horizon_days} días · objetivo de servicio {settings.target_probability:.1%} · "
            f"protección sin reposición {settings.protection_days} días.</div>",
            unsafe_allow_html=True,
        )

    selected_stock = stock[stock["Producto"] == selected_product].iloc[0]
    product_comparison = comparison[comparison["Producto"] == selected_product].copy()
    recommended = product_comparison[product_comparison["Recomendada"]]
    best = recommended.iloc[0] if not recommended.empty else product_comparison.iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Cobertura del stock definido",
        format_percent(selected_stock["Probabilidad_cobertura"]),
        delta=f"{selected_stock['Probabilidad_cobertura'] - settings.target_probability:+.1%} vs objetivo",
    )
    k2.metric(
        "Stock requerido",
        f"{selected_stock['Stock_recomendado_objetivo']:.0f} u.",
        delta=f"Brecha {selected_stock['Brecha_stock']:+.0f} u.",
        delta_color="normal",
    )
    k3.metric(
        f"Servicio con {best['Politica']}",
        format_percent(best["Nivel_servicio_unidades"]),
        delta=f"{best['Brecha_nivel_servicio']:+.1%} vs objetivo",
    )
    k4.metric(
        "Costo relevante esperado",
        format_currency(best["Costo_relevante_promedio"]),
        delta=f"CVaR 95% {format_currency(best['Costo_relevante_CVaR95'])}",
        delta_color="off",
    )

    criterion = (
        "cumple el objetivo y minimiza el costo relevante"
        if bool(best["Cumple_objetivo"])
        else "alcanza el mayor nivel de servicio entre los escenarios evaluados"
    )
    st.info(
        f"**Decisión sugerida:** aplicar **{POLICY_LABELS[best['Politica']]}** para "
        f"**{selected_product}**, porque {criterion}. El stock inicial es "
        f"**{selected_stock['Evaluacion'].lower()}** para el periodo de protección."
    )

    tab_summary, tab_risk, tab_cost, tab_detail = st.tabs([
        "Resumen ejecutivo",
        "Riesgo y variabilidad",
        "Costos y políticas",
        "Indicadores y trazabilidad",
    ])

    percentiles = results["percentiles_demanda"]
    chart_percentiles = percentiles[percentiles["Producto"] == selected_product].set_index("Percentil")
    trajectory_chart = trajectories[trajectories["Producto"] == selected_product].pivot(
        index="Dia", columns="Politica", values="Inventario_promedio"
    )

    with tab_summary:
        left_chart, right_chart = st.columns(2)
        with left_chart:
            st.markdown("#### Demanda acumulada por percentil")
            st.bar_chart(chart_percentiles["Demanda_acumulada"], color="#4d78a8", use_container_width=True)
            st.caption(
                "El percentil seleccionado traduce la incertidumbre de demanda en el stock "
                "necesario para el periodo de protección."
            )
        with right_chart:
            st.markdown("#### Frontera costo–servicio")
            st.scatter_chart(
                product_comparison,
                x="Costo_relevante_promedio",
                y="Nivel_servicio_unidades",
                color="Politica",
                size="Inventario_promedio",
                use_container_width=True,
            )
            st.caption("La alternativa dominante combina mayor servicio con menor costo controlable.")

        st.markdown("#### Trayectoria promedio del inventario disponible")
        st.line_chart(trajectory_chart, use_container_width=True)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Stock de seguridad sugerido", f"{selected_stock['Stock_seguridad_recomendado']:.0f} u.")
        r2.metric("Utilización esperada", format_percent(selected_stock["Utilizacion_stock"]))
        r3.metric("Rotación anualizada", f"{best['Rotacion_anualizada']:.1f} veces")
        r4.metric("Cobertura promedio", f"{best['Dias_cobertura_promedio']:.1f} días")

    with tab_risk:
        st.markdown("#### Riesgo del stock y cola extrema")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Probabilidad de quiebre", format_percent(selected_stock["Probabilidad_quiebre"]))
        q2.metric("Riesgo de sobrestock", format_percent(selected_stock["Probabilidad_sobrestock"]))
        q3.metric("CV de la demanda", format_percent(selected_stock["CV_demanda_periodo"]))
        q4.metric("Demanda CVaR 95%", f"{selected_stock['Demanda_CVaR95_periodo']:.0f} u.")

        q5, q6, q7, q8 = st.columns(4)
        q5.metric("Quiebre con política", format_percent(best["Probabilidad_quiebre_horizonte"]))
        q6.metric("Escenarios que logran objetivo", format_percent(best["Probabilidad_servicio_objetivo"]))
        q7.metric("Costo total CVaR 95%", format_currency(best["Costo_total_CVaR95"]))
        q8.metric("Variabilidad del costo", format_percent(best["CV_costo_total"]))

        risk_view = product_comparison[[
            "Politica", "Nivel_servicio_unidades", "Probabilidad_quiebre_horizonte",
            "Dias_sin_quiebre", "Unidades_no_atendidas", "Costo_relevante_VaR95",
            "Costo_relevante_CVaR95", "CV_costo_total",
        ]].copy()
        risk_view["Politica"] = risk_view["Politica"].map(POLICY_LABELS)
        st.dataframe(
            risk_view.style.format({
                "Nivel_servicio_unidades": "{:.2%}",
                "Probabilidad_quiebre_horizonte": "{:.2%}",
                "Dias_sin_quiebre": "{:.2%}",
                "Unidades_no_atendidas": "{:.1f}",
                "Costo_relevante_VaR95": "S/ {:,.2f}",
                "Costo_relevante_CVaR95": "S/ {:,.2f}",
                "CV_costo_total": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab_cost:
        st.markdown("#### Composición del costo esperado")
        cost_breakdown = product_comparison.set_index("Politica")[[
            "Costo_mantenimiento_promedio",
            "Costo_ordenamiento_promedio",
            "Costo_quiebre_promedio",
        ]]
        cost_breakdown.columns = ["Mantenimiento", "Ordenamiento", "Quiebre"]
        st.bar_chart(cost_breakdown, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Costo por unidad atendida", format_currency(best["Costo_por_unidad_atendida"]))
        c2.metric("Costo de mantenimiento", format_currency(best["Costo_mantenimiento_promedio"]))
        c3.metric("Costo de ordenar", format_currency(best["Costo_ordenamiento_promedio"]))
        c4.metric("Costo de quiebre", format_currency(best["Costo_quiebre_promedio"]))

        policy_view = product_comparison[[
            "Politica", "Nivel_servicio_unidades", "Inventario_promedio", "Ordenes_promedio",
            "Rotacion_anualizada", "Dias_cobertura_promedio", "Costo_por_unidad_atendida",
            "Costo_relevante_promedio", "Costo_total_promedio", "Recomendada",
        ]].copy()
        policy_view["Politica"] = policy_view["Politica"].map(POLICY_LABELS)
        st.dataframe(
            policy_view.style.format({
                "Nivel_servicio_unidades": "{:.2%}",
                "Inventario_promedio": "{:.1f}",
                "Ordenes_promedio": "{:.1f}",
                "Rotacion_anualizada": "{:.1f}",
                "Dias_cobertura_promedio": "{:.1f}",
                "Costo_por_unidad_atendida": "S/ {:,.2f}",
                "Costo_relevante_promedio": "S/ {:,.2f}",
                "Costo_total_promedio": "S/ {:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab_detail:
        definitions = pd.DataFrame([
            ("Cobertura del stock", "Probabilidad de que el stock inicial cubra toda la demanda del periodo sin reposición."),
            ("Fill rate", "Unidades atendidas inmediatamente entre unidades demandadas."),
            ("VaR 95%", "Umbral de costo o demanda que no se supera en aproximadamente 95% de los escenarios."),
            ("CVaR 95%", "Promedio de los resultados pertenecientes al 5% de escenarios más adversos."),
            ("Coeficiente de variación", "Desviación estándar dividida entre el promedio; mide variabilidad relativa."),
            ("Rotación anualizada", "Unidades atendidas respecto del inventario promedio, convertidas a una base anual."),
            ("Costo relevante", "Costo de mantenimiento + ordenamiento + quiebre; excluye compras para comparar políticas."),
            ("Frontera costo–servicio", "Relación entre costo controlable y nivel de servicio de cada política evaluada."),
        ], columns=["Indicador", "Definición operativa"])
        st.markdown("#### Definiciones de los indicadores")
        st.dataframe(definitions, use_container_width=True, hide_index=True)

        st.markdown("#### Resultado completo de validación del stock")
        stock_view = stock[stock["Producto"] == selected_product]
        st.dataframe(
            stock_view.style.format({
                "Probabilidad_cobertura": "{:.2%}",
                "Probabilidad_quiebre": "{:.2%}",
                "Probabilidad_sobrestock": "{:.2%}",
                "Fill_rate_sin_reposicion": "{:.2%}",
                "CV_demanda_periodo": "{:.2%}",
                "Utilizacion_stock": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "⬇️ Descargar resultados e indicadores avanzados en Excel",
        data=results_to_excel(results, settings),
        file_name="resultados_avanzados_montecarlo_inventarios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Herramienta de apoyo para decisiones. La calidad de la recomendacion depende de que "
    "la distribucion de demanda, los costos y el lead time representen adecuadamente la operacion."
)
