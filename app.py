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


st.title("📦 Simulacion Monte Carlo para Gestion de Inventarios")
st.caption(
    "Evalua si el stock determinado soporta la demanda estocastica y compara "
    "las politicas (Q,s), (T,S) y (s,S) bajo variabilidad de demanda y lead time."
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
    st.subheader("3. Resultados: ¿el stock determinado es suficiente?")
    selected_product = st.selectbox("Producto para el analisis visual", stock["Producto"].tolist())
    selected_stock = stock[stock["Producto"] == selected_product].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de cobertura", format_percent(selected_stock["Probabilidad_cobertura"]))
    m2.metric("Stock evaluado", f"{selected_stock['Stock_evaluado']:.0f} u.")
    m3.metric(
        f"Stock requerido al {settings.target_probability:.1%}",
        f"{selected_stock['Stock_recomendado_objetivo']:.0f} u.",
        delta=f"Brecha {selected_stock['Brecha_stock']:+.0f} u.",
        delta_color="normal",
    )
    m4.metric("Diagnostico", selected_stock["Evaluacion"])

    if selected_stock["Evaluacion"] == "Suficiente":
        st.success(
            f"El stock de {selected_stock['Stock_evaluado']:.0f} unidades alcanza o supera "
            f"el nivel objetivo durante {settings.protection_days} dias."
        )
    else:
        st.warning(
            f"El stock tiene una probabilidad de cobertura de "
            f"{selected_stock['Probabilidad_cobertura']:.1%}. Para alcanzar el objetivo de "
            f"{settings.target_probability:.1%}, el modelo estima "
            f"{selected_stock['Stock_recomendado_objetivo']:.0f} unidades."
        )

    st.dataframe(
        stock.style.format({
            "Probabilidad_cobertura": "{:.1%}",
            "Fill_rate_sin_reposicion": "{:.1%}",
            "Demanda_promedio_periodo": "{:.1f}",
            "Demanda_p95_periodo": "{:.1f}",
            "Dia_promedio_quiebre": "{:.1f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    percentiles = results["percentiles_demanda"]
    chart_percentiles = percentiles[percentiles["Producto"] == selected_product].set_index("Percentil")
    st.caption("Demanda acumulada simulada por percentil")
    st.bar_chart(chart_percentiles["Demanda_acumulada"], use_container_width=True)

    st.subheader("4. Comparacion de politicas y costos")
    product_comparison = comparison[comparison["Producto"] == selected_product].copy()
    recommended = product_comparison[product_comparison["Recomendada"]]
    if not recommended.empty:
        best = recommended.iloc[0]
        criterion = "cumple el nivel objetivo y minimiza el costo relevante" if best["Cumple_objetivo"] else "logra el mayor servicio entre los escenarios evaluados"
        st.info(
            f"Politica recomendada para **{selected_product}: {POLICY_LABELS[best['Politica']]}**, "
            f"porque {criterion}. Costo relevante promedio: "
            f"**{format_currency(best['Costo_relevante_promedio'])}**."
        )

    display_comparison = product_comparison.copy()
    display_comparison["Politica"] = display_comparison["Politica"].map(POLICY_LABELS)
    st.dataframe(
        display_comparison.style.format({
            "Nivel_servicio_unidades": "{:.2%}",
            "Dias_sin_quiebre": "{:.2%}",
            "Probabilidad_servicio_objetivo": "{:.2%}",
            "Inventario_promedio": "{:.1f}",
            "Unidades_no_atendidas": "{:.1f}",
            "Eventos_quiebre": "{:.1f}",
            "Ordenes_promedio": "{:.1f}",
            "Costo_mantenimiento_promedio": "S/ {:,.2f}",
            "Costo_ordenamiento_promedio": "S/ {:,.2f}",
            "Costo_quiebre_promedio": "S/ {:,.2f}",
            "Costo_compras_promedio": "S/ {:,.2f}",
            "Costo_relevante_promedio": "S/ {:,.2f}",
            "Costo_total_promedio": "S/ {:,.2f}",
            "Costo_total_p95": "S/ {:,.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.caption("Nivel de servicio por politica")
        service_chart = product_comparison.set_index("Politica")[["Nivel_servicio_unidades"]]
        st.bar_chart(service_chart, use_container_width=True)
    with chart_col2:
        st.caption("Costo relevante promedio por politica")
        cost_chart = product_comparison.set_index("Politica")[["Costo_relevante_promedio"]]
        st.bar_chart(cost_chart, use_container_width=True)

    st.caption("Trayectoria promedio del inventario disponible")
    trajectory_chart = trajectories[trajectories["Producto"] == selected_product].pivot(
        index="Dia", columns="Politica", values="Inventario_promedio"
    )
    st.line_chart(trajectory_chart, use_container_width=True)

    st.download_button(
        "⬇️ Descargar resultados completos en Excel",
        data=results_to_excel(results, settings),
        file_name="resultados_montecarlo_inventarios.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Herramienta de apoyo para decisiones. La calidad de la recomendacion depende de que "
    "la distribucion de demanda, los costos y el lead time representen adecuadamente la operacion."
)
