# Research inicial sustituido durante el cierre

Este artefacto se conserva como evidencia inicial y no es la release activa.
Se sustituyó por research_risk_v1_1 al comprobar que el conteo vectorizado de
XGBoost comparaba float32 contra umbrales reducidos a float32. Dos registros
de desarrollo pasaban de HIGH a MEDIUM bajo la frontera decimal exacta servida
por la API. Scores, límites derivados y métricas de junio no cambiaron.

La implementación final comparte una sola regla en dominio entre análisis y
serving; v1_1 incluye los conteos/intervalos corregidos y un nuevo hash. La versión
research_risk_v1 ya no es aceptada por el loader final. No se modificaron su JSON
ni su registro de publicación inicial, ni se promovió a aprobación operativa.
