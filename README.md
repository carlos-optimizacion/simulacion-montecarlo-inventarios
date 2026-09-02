# Simulacion Monte Carlo para gestion de inventarios

Aplicacion en Streamlit para comprobar si un stock determinado puede soportar
la variabilidad de la demanda y comparar politicas de reposicion bajo demanda y
lead time estocasticos.

## Aplicacion publicada

Use la aplicacion en:

https://simulacion-montecarlo-inventarios-nqrfxjxwbvddkxv8fadftb.streamlit.app/

## Funciones

- Ingreso manual de uno o varios productos.
- Importacion desde Excel mediante una plantilla descargable.
- Demanda Normal, Poisson, Triangular o empirica/historica.
- Validacion probabilistica del stock durante un periodo de proteccion.
- Comparacion de politicas `(Q,s)`, `(T,S)` y `(s,S)`.
- Nivel de servicio por unidades, dias sin quiebre, inventario promedio,
  unidades no atendidas, numero de ordenes y eventos de quiebre.
- Costos de mantenimiento, ordenamiento, quiebre, compras y costo total.
- Recomendacion de la politica de menor costo que cumple el nivel de servicio.
- Exportacion de todos los resultados a Excel.

## Ejecucion local

1. Instale Python 3.10 o superior.
2. Abra una terminal en esta carpeta.
3. Cree un entorno e instale dependencias:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS o Linux
source .venv/bin/activate
pip install -r requirements.txt
```

4. Ejecute la aplicacion:

```bash
streamlit run app.py
```

## Publicacion gratuita en Streamlit Community Cloud

1. Suba los archivos a un repositorio de GitHub.
2. Ingrese a https://share.streamlit.io/ y vincule su cuenta.
3. Seleccione el repositorio y configure `app.py` como archivo principal.
4. Pulse **Deploy**.

## Estructura del Excel

La aplicacion genera una plantilla con tres hojas:

- `Productos`: parametros de demanda, costos y politicas.
- `Demanda_historica`: observaciones por producto para la distribucion empirica.
- `Diccionario`: definicion de cada campo.

## Criterio de recomendacion

Para cada producto, la herramienta selecciona la politica con el menor costo
relevante promedio entre las que alcanzan el nivel de servicio objetivo. Si
ninguna lo alcanza, destaca el escenario con mejor nivel de servicio y, ante un
empate, menor costo relevante.

El costo relevante es la suma de mantenimiento, ordenamiento y quiebre. El
costo total tambien incorpora las compras realizadas durante el horizonte.

## Supuestos actuales

- Se modelan ventas perdidas; no se acumulan pedidos pendientes.
- Los pedidos se colocan al final del dia y llegan al inicio del dia de arribo.
- El lead time sigue una Normal truncada a un minimo de un dia.
- La posicion de inventario es inventario disponible mas inventario en transito.
- Los resultados representan promedios de multiples repeticiones Monte Carlo.
