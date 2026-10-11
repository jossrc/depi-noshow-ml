# Actualización futura de datos y modelos

Estado: **procedimiento documentado, no ejecutado**. FLOWww/MySQL no se contactó.
La ejecución del 11-10-2026 solo autorizó SELECT sobre PostgreSQL existente;
actualizar origen, réplica o analytics requiere una autorización posterior.
Este documento no define endpoints o credenciales externos inexistentes en Python.

## Responsabilidades y secuencia

1. El responsable externo de FLOWww/ETL debe acordar alcance, destino, permisos,
   respaldo y recuperación antes de sincronizar. La ingesta trae cambios del
   origen; actualizar una réplica y transformar analytics son operaciones
   distintas. FastAPI de inferencia no realiza ninguna de ellas.
2. Conservar eventos y revisiones antes de reconstruir un dataset. Capturar ID
   original, ID de copia/revisión, relación anterior–siguiente, creación,
   modificación, fecha programada, motivo de cambio, cancelación/reprogramación,
   detalles de servicios por revisión y momento en que cada evento fue conocido.
   Separar `event_at`, marca registrada, `known_at`, ingesta y procesamiento.
   Una edición retroactiva no debe fingir disponibilidad anterior.
3. Versionar el catálogo y los atributos mutables de reserva/cliente utilizados
   por las 19 features. Conservar una instantánea al crear la reserva; las líneas
   SCHEDULED actuales y `computed_at` no acreditan ese snapshot.
4. Reconstruir analytics fuera de este servicio Python con reglas versionadas.
   Validar enlaces y detalles antes de deduplicar: la auditoría local encontró
   24 enlaces sin cabecera destino y 114 detalles huérfanos. Un enlace a una
   cabecera final no garantiza todas las revisiones intermedias.
5. Documentar elegibilidad histórica y definición de NO_SHOW_CANDIDATE. El flag
   `training_eligible` actual es false incluso en las 213.873 filas del dataset
   auditado. No reconstruir población o cambiar target usando solo ese flag.
   CANCELLED y RESCHEDULED_HISTORY no se convierten en no-shows.
6. Después de recibir una fuente reconstruida y validada, Python puede exportar
   en REPEATABLE READ READ ONLY a una **ruta nueva**. No reutilizar CSV, manifest,
   auxiliar ni directorio de experimento anteriores. El exportador principal
   tiene soporte histórico de reemplazo: el operador debe elegir y comprobar un
   destino inexistente antes de invocarlo. No ejecutarlo sobre el original.
7. Generar manifest/hash y auxiliar de disponibilidad coherentes con la misma
   fuente. `export-label-availability` comprueba exactamente los bytes del dataset
   en PostgreSQL antes de publicar; si la fuente cambió, reconstruir un par nuevo
   legítimo, sin modificar hashes para eludir la comprobación.
8. Auditar integridad, snapshots, historial as-of, disponibilidad de etiquetas,
   madurez, zona Lima, convención de domingo y sesgo de selección. Conservar
   advertencias de cierres manuales y bloquear anomalías críticas.
9. Registrar un protocolo reproducible antes de entrenar: identidad de fuentes,
   revisión con evidencia/responsable, semilla, cortes por `prediction_at`, purga
   por `label_recorded_at`, dependencias y criterios. Preprocesamiento ajustado
   solo en train. Mantener exactamente las 19 predictoras hasta una decisión
   explícita de versionado; capturar metadatos adicionales no los hace features.
10. Evaluar RF y XGB independientes en validación. Calibradores o umbrales
    necesitan datos temporales independientes y costes/capacidad definidos.
    El experimento exploratorio actual no puede abrir su test reservado mediante
    `evaluate`; no falsificar revisión ni promoverlo retrospectivamente.
11. Revisar y aprobar una nueva release explícita. Preservar versiones anteriores,
    pines confiables, hashes y resolución de dependencias. Readiness exige ambos
    modelos; probar single/compare/batch/SHAP con originales/candidatos aprobados.
    Cambiar referencia o catálogo requiere un proceso versionado; no sustituir
    archivos bajo un proceso ya cargado.

## Verificaciones antes de una actualización

- Confirmar autorización futura de escritura y responsables externos; revisar
  respaldo, recuperación y límites de infraestructura sin asumir permiso actual.
- Resolver duplicados por revisión sin confundir reserva vigente con archivo
  histórico; conservar cada detalle y su vigencia/known_at.
- Comprobar conflictos de clase, reglas y cambios retroactivos. Mantener la
  población elegible y exclusiones explícitas, con conteos por período/sede.
- Registrar versión de reglas y disponibilidad real: cierre o tracking actuales
  no bastan para conocer qué estaba disponible en una fecha anterior.
- Auditar contaminación por etiquetas inmaduras y selección diferencial por
  target. Conservar el test fuera del ajuste, calibración y selección.
- Guardar manifiestos nuevos y verificar hashes antes/después. No copiar `.env`,
  datos individuales ni credenciales a reportes o Git.
- Comparar población antes/después de la transición comercial aproximada
  junio–julio 2026; no inventar un día exacto ni atribuir toda deriva a esa política.
- Acordar aprobación/piloto y rollback de datos y release por separado. Volver al
  modelo anterior no deshace una ingesta ni reconstruye información perdida.
- La sincronización y el procesamiento de analytics siguen fuera del repositorio
  Python de serving. No se incorporó un sincronizador externo en esta tarea.

Los comandos locales de análisis y entrenamiento están en [phase2.md](phase2.md).
La integración futura está en [python_to_backend_handoff.md](python_to_backend_handoff.md).
