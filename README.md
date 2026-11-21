# Datos SINTÉTICOS de práctica – Proyecto Final CCI (AndesVision AI)

Sirven para desarrollar y depurar el notebook mientras llegan los archivos reales del profe.
**Ninguna cifra de estos datos va al informe.** Cuando lleguen los reales, solo cambias la ruta y haces Run All.

## Archivos
- solicitudes.csv: ventana de 1200 s, mismo esquema que el real (~18.5 mil filas, con filas "sucias").
- monitoreo.csv: 6000 minutos con D1, D2, incidente.
- solicitudes_validacion.csv: otra ventana, limpia; úsala SOLO en F6.
- generador_datos_sinteticos.py: código que los crea (semilla de fabricación 777, distinta de tu S).
- CLAVE_respuestas_NO_ABRIR_hasta_terminar_A1.json: lista de las inconsistencias inyectadas, para comprobar tu auditoría A1 después de hacerla.

## Qué traen incorporado (para que cada ítem tenga algo que encontrar)
- A: tramo anómalo (aprox. 780–960 s) con más tráfico, GPU más lenta y más fallos; filas inválidas, duplicadas, NaN, etc.
- B: llegadas Poisson por tramos; intentos geométrica truncada en 3 (abandono); versión B con menos fallos.
- C: G = 0 si hay caché (mezcla discreta-continua); parte continua gamma; red_ms lognormal (no normal).
- D: caché depende de la clase de tamaño (no independientes); E[G | tam] lineal con varianza creciente (heterocedasticidad).
- E: D1 y D2 con un factor común (no condicionalmente independientes); prevalencia en pruebas ≈ 12–16 %, no 0,6 %.
- F: versión B ligeramente más rápida; la ventana de validación tiene parámetros algo distintos (GPU más lenta, más fallos).

## Datasets públicos reales (para tener contraste real, requieren descarga manual)
- Azure Public Dataset (github.com/Azure/AzurePublicDataset): trazas de invocaciones de Azure Functions con instantes de llegada y finalización; sirve para la parte de llegadas y tiempos de servicio.
- Azure LLM Inference traces (2023/2024): tokens de entrada/salida por solicitud; el tamaño de la entrada puede hacer de proxy de tam_kb.
- BurstGPT (github.com/hpmll/burstgpt): instantes de solicitud, tokens de petición y respuesta; buena para llegadas con ráfagas.
Ninguno trae exactamente el esquema del profe, así que necesitarían un mapeo (renombrar columnas, inventar caché/versión). Por eso lo sintético es la mejor opción para probar el pipeline completo.
