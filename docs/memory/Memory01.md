# TESIS 2026–2027 — Memoria técnica integral y bitácora de decisiones

> **Versión de esta memoria:** 1.0 — 8 de octubre de 2026 (America/Lima).  
> **Proyecto:** *Sistema web predictivo basado en XGBoost y Random Forest para el pronóstico de inasistencias en una empresa del rubro de estética y cuidado personal*.  
> **Rol:** documento de continuidad/handoff para futuras conversaciones, Codex, desarrollo en 2026–2027, implementación de APIs, investigación y sustentación.  
> **Repositorio Python auditado:** <https://github.com/jossrc/depi-noshow-ml>, rama `main`, commit `77ea0fe6f8422c152412bda863aef6b8dbabab1b` al elaborar esta versión.  
> **Estado general:** Fases 1 y 2 Python implementadas; datos auditados; modelos exploratorios entrenados y optimizados; **sin modelo científicamente aprobado, sin evaluación final de prueba, sin calibración operativa y sin FastAPI implementada**.  
> **Privacidad:** en el documento académico se anonimiza la empresa; aquí se usa «DEPI» como alias operativo interno. No incorporar credenciales, datos personales de clientes, ni volcados completos de datos en Git.

---

## 0. Leer primero: resumen ejecutivo e instrucciones para retomar

### 0.1 Qué existe hoy

1. Existe una réplica de información relevante del sistema operativo **FLOWww**, cuyo origen es **MySQL 5**, en PostgreSQL local **18.4**, separada en esquemas `raw_flow` y `analytics`. Se contempla un esquema `ml` en la arquitectura de datos futura; **no debe interpretarse como prueba de que sus tablas de inferencia ya estén implementadas**.
2. En PostgreSQL se construyó `analytics.appointment_training_dataset_v1` mediante SQL, con 27 columnas: **19 predictoras** más 8 campos de identificación, fechas, etiqueta, auditoría y metadatos. La instantánea exportada contiene **213.873 citas**: 169.553 asistencias (`target=0`) y 44.320 candidatos a inasistencia (`target=1`).
3. El repositorio `jossrc/depi-noshow-ml` incluye exportación de solo lectura, contratos, auditorías, exportación auxiliar de fechas de disponibilidad de etiquetas, división cronológica, entrenamiento, evaluación potencial, 12 experimentos exploratorios iniciales y una única ronda de optimización de 16 candidatos adicionales.
4. En validación (reservas abril–junio de 2026), el **Random Forest completo** obtuvo **ROC-AUC 0,7502**, **AP 0,5126** y **recall 0,5091** a umbral diagnóstico 0,5. El **XGBoost completo** obtuvo **ROC-AUC 0,7247**, **AP 0,4731** y **recall 0,4177**. En una ronda posterior, el XGBoost completo `xgb_unweighted` alcanzó **ROC-AUC 0,7278**, **AP 0,4832**, pero recall **0,0833** al mismo umbral. **Random Forest es el candidato provisional, no el ganador definitivo**.
5. El documento PI1 v4 fija una meta de **ROC-AUC ≥ 0,80 en prueba**. **No se ha cumplido** con lo observado en validación; **la prueba final no se ha ejecutado**. Las métricas de ROC-AUC no son porcentajes de exactitud.
6. El entrenamiento realizado es expresamente **`EXPLORATORY_NOT_VALIDATED`**. Hay **ocho controles metodológicos** todavía pendientes de aprobación humana. No se deben marcar `verified` sin evidencia. La prueba temporal reservada comprende reservas de julio–septiembre de 2026 y sigue protegida.
7. Se identificó un **cambio comercial aproximado entre junio y julio de 2026**: antes podían reservar y pagar al llegar; después se exige en general adquirir previamente un bono/voucher para reservar, con excepción de atención presencial en el mismo día si hay cupo. **No hay día exacto confirmado ni efecto causal exclusivo demostrado.**
8. El siguiente desarrollo priorizado es **FastAPI**, para exponer el pipeline predictivo provisional de manera aislada, seguido de la integración con **Node.js/Fastify** y **Next.js/React**. No se debe anunciar producción ni bandas de riesgo validadas hasta la evaluación científica/operativa.

### 0.2 Principios de trabajo acordados

- **Paso a paso:** el usuario prefiere una instrucción o decisión concreta por iteración y que se interprete el resultado antes de continuar. Evitar nuevas auditorías redundantes.
- **No rehacer la base por precaución genérica:** mantener el CSV original, el auxiliar y los hashes. No ejecutar nuevamente ETL o borrados sin una justificación concreta y respaldo.
- **Diferenciar evidencia:** «confirmado por consultas/archivos», «regla de negocio informada», «resultado reportado por Codex», «hipótesis/limitación», «pendiente».
- **No confundir candidato a no-show con ausencia individual verificada**. Ni una cancelación con una inasistencia, ni un ticket de S/ 0 con una ausencia, ni un bono actual con un dato histórico.
- **No filtrar la prueba para escoger hiperparámetros**: conservar la separación temporal y los artefactos originales.
- **Seguridad:** nunca subir `.env`, CSV reales con IDs, modelos locales, credenciales o información personal a un repositorio público. Para futuras conversaciones compartir estadísticas y manifiestos, no identificadores individuales.

### 0.3 Leyenda de certeza

- **[VERIFICADO SQL/ARCHIVO]**: consta en scripts, JSON/CSV, capturas, cálculos/consultas ya compartidas o código examinado.
- **[REPORTADO CODEX]**: resumido por Codex al ejecutar localmente; el código fue revisado en GitHub, pero no se reprodujeron aquí los entrenamientos ni se inspeccionaron sus modelos binarios.
- **[NEGOCIO]**: descripción directa del comportamiento de FLOWww proporcionada por quien conoce la operación.
- **[HIPÓTESIS]**: explicación plausible, no demostración causal.
- **[PENDIENTE]**: requiere validación, evidencia, aprobación, integración o revisión futura.

---

## 1. Identidad académica, alcance y criterios de éxito

### 1.1 Documento académico de referencia

- Archivo original analizado: **`PI1 Trabajo Investigación v4.docx`**, fechado el **3 de octubre de 2026** en la portada.
- Titulación: trabajo de investigación del programa de Ingeniería de Sistemas de la UPC; **no confundir este documento de PI1 con resultados finales de PI2**.
- Contexto: empresa del rubro de estética y cuidado personal, principalmente servicios de depilación láser, con múltiples sedes en Lima; el área de intervención es **Central de Operaciones / Call Center**, dentro de la gestión de recepción y agendamiento.
- Problema: reservas no concretadas que producen capacidad ociosa (especialistas, cabinas, equipos), incertidumbre operativa y pérdidas de oportunidades.
- Objetivo general: desarrollar un sistema web predictivo basado en XGBoost y Random Forest para estimar anticipadamente el riesgo de inasistencia y apoyar la gestión de la agenda.
- Objetivos específicos, según PI1: (1) analizar/seleccionar tecnologías y algoritmos; (2) diseñar arquitectura, componentes, flujos e interfaces; (3) desarrollar la solución web integrando ambos algoritmos; (4) validar rendimiento predictivo y funcionalidad/usabilidad con métricas y criterios de aceptación.

### 1.2 Indicadores de éxito que SÍ figuran en PI1 v4

| Indicador / lugar | Criterio establecido | Estado al 8-10-2026 |
|---|---|---|
| **IND-04**, sección «Indicadores de Éxito» | Validar desempeño predictivo y usabilidad. Menciona **área bajo la curva**, **sensibilidad/recall**, **F1** y **usabilidad**. No se encontró un mínimo numérico explícito de recall o F1 en esa fila. | Comparación exploratoria realizada; usabilidad del sistema web pendiente. |
| **Línea base de calidad**, anexo de calidad | **AUC ROC ≥ 0,80**, medido **en conjunto de prueba**. | **No acreditado**. Mejor ROC-AUC de la comparación principal ≈ 0,75 **en validación**, no en prueba. |
| **Riesgos**, anexo | «AUC < 0,70» aparece como **ejemplo** de desempeño insuficiente que dispararía una respuesta. | No constituye un segundo umbral de aprobación ni sustituye la meta 0,80. |
| Otros compromisos | Arquitectura, historias de usuario aceptadas, rendimiento del API (línea base: respuesta promedio ≤ 5 segundos), etc. | Se evalúan por separado del desempeño predictivo. |

**Regla académica esencial:** una investigación metodológicamente válida puede documentar que no se alcanzó una meta. **No bajar retrospectivamente AUC ≥ 0,80 solo para declarar éxito**. Si el asesor aprueba una revisión del umbral, dejar constancia de motivo, versión, fecha, fundamento y distinción entre evaluación exploratoria y prueba final. No publicar «0,75 de accuracy»: 0,75 aquí es **ROC-AUC**, una medida de discriminación independiente del umbral.

### 1.3 Qué se pretende entregar en el sistema

- Riesgo estimado de no-show **antes de que ocurra la cita** a partir de información conocida en el instante de predicción.
- Interfaz comprensible para personal operativo, con reservas, probabilidades/niveles de riesgo (solo tras calibración y validación de las bandas) y factores asociados a predicciones, sujeto a interpretación responsable.
- Funciones futuras: priorizar confirmaciones o recordatorios, consulta de agenda, soporte a reasignaciones y decisiones operativas. **No se ha probado que el sistema reduzca realmente las inasistencias, costos o espacios ociosos**; ese impacto requerirá evaluación prospectiva/controlada.
- En PI1 se propone una validación cuantitativa predictiva con datos históricos y validación operativa de la herramienta mediante juicio de expertos. No confundir la descripción de lo planeado con resultados ya obtenidos.

---

## 2. Arquitectura general: negocio, datos, ML y software

### 2.1 Arquitectura objetivo de software

```text
[Central de Operaciones / navegador]
                 |
                 v
         [Frontend Next.js / React]
                 |
             HTTP / JSON
                 v
         [Backend Node.js / Fastify]
          |             |
          |             +----> [Microservicio Python / FastAPI]
          |                         |
          |                         +---> pipeline de preprocesamiento
          |                         +---> modelo versionado RF o XGBoost
          |                         +---> probabilidad, versión, controles
          |
          +----> [PostgreSQL / capa de consulta y features]

[FLOWww MySQL 5] -> [replicación/cargas] -> [PostgreSQL 18.4]
                                        ├── raw_flow (copia operativa)
                                        ├── analytics (ETL, etiquetas, features)
                                        └── ml (espacio lógico previsto para inferencia)
```

**Responsabilidades previstas:**

- **Next.js/React:** interfaz web, autenticación de sesiones y vistas de agenda/predicción (a implementar).
- **Node.js/Fastify:** API principal, autenticación/autorización, lectura de reservas y cálculo de variables «as-of» para una petición de predicción, composición/orquestación de solicitudes al microservicio Python, reglas de negocio y registro trazable (a implementar).
- **Python/FastAPI:** exposición de una interfaz de inferencia estable que cargue el modelo y **el mismo pipeline** que se utilizó en entrenamiento; no volver a calcular las 19 features desde un CSV de entrenamiento ni introducir etiquetas reales en inferencia (a implementar).
- **PostgreSQL:** copia, estructuras analíticas, catálogo, resultados, historial temporal, eventual persistencia de scores si se aprueba (ya están las tablas analíticas principales; otras son futuras).

La arquitectura de software aparece en PI1 (apartado 3.3, Arquitectura Técnica Base/Específica) y en decisiones tomadas a lo largo del proyecto. **Lo planificado no equivale a endpoints ya implementados.**

### 2.2 Arquitectura de datos

```text
FUENTE OPERATIVA
FLOWww / MySQL 5
    |
    | migración histórica inicial; sincronización futura principalmente por CSV
    v
PostgreSQL 18.4 local (Docker: servicio postgres; contenedor depi_noshow)
    |
    +-- raw_flow: copia de entidades, IDs, nombres y marcas originales
    |
    +-- analytics:
    |    v_appointment_outcomes_v1
    |    appointment_outcomes
    |    appointment_events
    |    appointment_services
    |    client_history_current                  [NO usar para retrospectivo]
    |    product_catalog_resolution             [snapshot histórico no acreditado]
    |    ml_history_cumulative_v1               [historial ordenado por known_at]
    |    appointment_training_dataset_v1        [fuente oficial del CSV]
    |
    +-- ml: namespace previsto / estado de despliegue por definir
          |
          v
 [Extracción Python READ ONLY y CSV local]
          |
          v
 [Auditoría -> entrenamiento -> evaluación -> servicio FastAPI futuro]
```

- **La base PostgreSQL es fuente de verdad del dataset analítico; el CSV es una exportación vinculada al snapshot.**
- La carga a PostgreSQL **ya ocurrió**, no hay que volver a «migrar desde cero».
- Para entorno desplegado se contemplan cargas idempotentes desde CSV; **la estrategia concreta de actualizaciones, CDC, snapshots/deltas y borrados aún debe resolverse**, en especial porque ciertos servicios se borran físicamente en FLOWww.
- No consultar la base productiva MySQL en cada predicción sin rediseño y autorizaciones; el sistema se diseñó para independencia operativa y minimizar carga.
- El contenedor documentado en el repo es `depi_noshow`, la base predeterminada `depi-noshow-db`, usuario predeterminado `admin`, puerto publicado solo en `127.0.0.1` desde Compose; **verificar configuración real del volumen antes de conectar**. Hay documentos históricos que escriben `depi_noshow` como nombre de base/entorno: no dar por hecho que coincidan con la configuración actual.

### 2.3 Entornos y stack actual de ML

- Python **3.12+**, paquete instalable `depi-noshow-ml`, CLI `depi-ml`.
- psycopg 3, python-dotenv, pandas, NumPy, scikit-learn, XGBoost, matplotlib, joblib; SHAP opcional.
- `pytest` para tests; Git/GitHub como trazabilidad de código; artefactos locales ignorados en Git.
- `docker-compose.yml` define PostgreSQL **18.4** con almacenamiento persistente existente y healthcheck. **No ejecutar `docker compose down -v` ni borrar el directorio del volumen**.
- El esquema fuente es `raw_flow`; el principal analítico es `analytics`; huso de negocio **America/Lima** y timestamps conscientes de zona en Python/CSV (UTC en el archivo).

---

## 3. FLOWww: tablas, relaciones y reglas de negocio

### 3.1 Modelo operativo de entidades

| Tabla FLOWww (`raw_flow`) | Propósito / campos orientativos | Precaución |
|---|---|---|
| `clients` | Clientes; `ClientID`, `ClientBirthDate`, `ClientSex`, antigüedad y atributos registrados. | Posibles valores por defecto, datos editados y privacidad. |
| `diary_gen` | Cabecera de cita; `DiaryGID`, `DiaryGClientID`, `DiaryGClinicID`, `DiaryGCreated`, `DiaryGStart`, `DiaryGEnd`, `DiaryGLaserGID`, `DiaryGNotAttended`, `DiaryGDelayed`, `DiaryGDelayedID`, `DiaryGLocked`, `DiaryGTrackingStamp`, `DiaryGTrackingUID`, `DiaryGTags`, `DiaryGFWA`. | Campo «NotAttended» por sí solo **no demuestra** no-show. |
| `diary_det` | Servicios **programados**; `DiaryID`, `DiaryGID`, `DiaryZoneID`, `DiaryVoucherID`, sesión/pack. | Se pueden editar; los detalles históricos se gestionan conforme a reprogramaciones/copia y deben comprobarse. |
| `laser_gen` | Cabecera de sesión/atención; `LaserGID`, `LaserGCloseDate`, `LaserGTicketGID`. | Normalmente se crea el día de atención; cierre **manual** de especialista. |
| `laser_det` | Servicios **realmente atendidos**. | No son predictores permitidos en el instante de reservar. |
| `tickets_gen` / `tickets_det` | Ticket y detalle; `TicketGClosed`, compras/prestaciones. | Ticket cerrado es fuerte evidencia operativa de atención, pero no incluye timestamp histórico exacto de servicio en todos los casos. Ticket de valor 0 puede corresponder a bono. |
| `vouchers` | Bonos, compras, consumos y saldo, p. ej. `VoucherAvailable`, `VoucherExpiryDate`. | **Saldo actual no es saldo histórico**; no utilizar retrospectivamente sin snapshot. |
| `x_config_products_det` | Catálogo de productos, zonas, servicios, características médicas. | Cambios actuales del catálogo pueden alterar la reconstrucción histórica. |
| `x_config_clinics` | Catálogo de sedes. | Identidades/categorías requieren control de cambios. |
| `__x_config_users_view` / `__x_config_workplaces_view` | Personal y puestos. | No hay features individuales de especialistas en v1. |
| `ingestion_batches` | Seguimiento de lotes de importación. | Importante para futura sincronización y reproducibilidad. |

**Relaciones principales:**

```text
clients.ClientID             -> diary_gen.DiaryGClientID
 diary_gen.DiaryGID          -> diary_det.DiaryGID
 diary_gen.DiaryGLaserGID    -> laser_gen.LaserGID
 laser_gen.LaserGTicketGID   -> tickets_gen.TicketGID
 tickets_gen.TicketGID       -> tickets_det.TicketGID
 diary_det.DiaryZoneID       -> x_config_products_det.ProductID
```

**Separar tres nociones:** `diary_det` = lo reservado; `laser_det` = lo efectivamente atendido; `tickets_det` = lo facturado/cobrado. Las tres pueden diferir en el mismo episodio: cliente puede solicitar otra zona al acudir.

### 3.2 Flujo de una cita y reprogramaciones [NEGOCIO]

1. Se agenda una cita (`diary_gen`) y se asignan servicios previstos (`diary_det`).
2. El cliente puede confirmar, cancelar, editar servicios o reprogramar.
3. **Corrección importante de una interpretación inicial:** el usuario aclaró después que editar servicios **sí genera un nuevo registro/copia asociado al mecanismo de reprogramación**, incluso cuando la duración aumenta, disminuye o se mantiene. La modificación de detalles puede borrar físicamente filas en el registro operativo. No conservar como cierta la primera interpretación de que «solo se borra físicamente sin nueva cita».
4. `diary_gen.DiaryGDelayed = -1` identifica la copia histórica desplazada, y `DiaryGDelayedID` apunta a la cita vigente/destino según el ETL implementado; SQL `03_analytics_appointment_events.sql` traduce esa relación en `analytics.appointment_events` de tipo `RESCHEDULED`.
5. Al iniciar la atención, normalmente se genera `laser_gen`; los servicios realmente prestados se registran en `laser_det`. Pueden diferir de lo originalmente reservado.
6. La especialista/personal cierra **manualmente** la sesión; la fecha `LaserGCloseDate` es **registro administrativo**, no prueba de hora física exacta de prestación.

**Resultado auditado:** consulta sobre 162.905 citas con eventos de reprogramación detectada: **0** incluidas en `appointment_training_dataset_v1`; 143.365 con outcome `ATTENDED_COMPLETED` o `NO_SHOW` en `appointment_outcomes`. Esto confirma exclusión de reprogramaciones **detectadas**, no cobertura absolutamente completa de todas las ediciones.

### 3.3 Tags de agenda, confirmaciones y separaciones críticas

- `DiaryGTags` es una cadena de IDs de tag separados por `|`. Un ejemplo conocido: `-1|16|15`.
- **Tag `-1` ≠ `DiaryGTrackingUID = -1`.** El primero corresponde a confirmación WhatsApp, según el proceso informado; el segundo indica origen/usuario del tracking en la clasificación de resultados y requiere cautela.
- **Tag `16`** = borrador de cita, exclusión estructural `DRAFT`.
- **Tag `15`** = etiqueta histórica de «cliente cancelador», aplicada masivamente/no fiable para predictor.
- **Tag `8`** = cliente en sede, no equivale automáticamente a atención completada.
- `DiaryGConfirmed` figura en el DDL original como confirmación por correo; no equipararlo con el tag WhatsApp sin comprobar operación real.
- Proceso de WhatsApp conocido: aviso cercano a **tres días** antes de la cita solicitando confirmación con botón; generalmente se envían **dos mensajes**. No se confirmó el instante exacto del segundo. La etiqueta de confirmación final puede haberse incorporado después de la creación de cita; no se usa como predictor retrospectivo sin historial de timestamps.
- La coincidencia de tags debe ser por **token exacto**, no `LIKE '%-1%'` (podría coincidir con IDs mayores).

### 3.4 Regla de cancelación e inasistencia

La columna `DiaryGNotAttended = -1` se utiliza operativamente para una **marca de cancelación** y no basta para afirmar inasistencia. El ETL de outcomes combina: marca de cancelación, seguimiento/usuario/fecha, ticket cerrado, sesión, fecha de la cita y exclusiones. Por eso `target=1` es **«candidato no-show» según reglas FLOWww**, no constatación humana individual de cada evento. Mantener la distinción en texto de tesis, APIs y dashboards.

Los estados `analytics.appointment_outcomes.outcome` definidos como ENUM son:

`ATTENDED_COMPLETED`, `NO_SHOW`, `CANCELLED`, `RESCHEDULED_HISTORY`, `IN_PROGRESS`, `FUTURE`, `BLOCKED`, `DRAFT`, `IGNORED`, `UNKNOWN`.

Detalles (`classification_detail`): `ATTENDED`, `NO_SHOW_CANDIDATE`, `NO_SHOW_REVIEW_SYSTEM`, `CANCELLED_OR_UNCLASSIFIED`, `FUTURE_CANCELLED`, `FUTURE_PENDING`, `SESSION_IN_PROGRESS`, `PAST_UNRESOLVED`, `REVIEW_CONFLICT`, `EXCLUDED`.

- `ATTENDED_COMPLETED`: atención completada según reglas y ticket cerrado, no basta con visita indicada o importe monetario positivo.
- `NO_SHOW`: resultado derivado de candidatos no-show con condiciones de tracking; `NO_SHOW_REVIEW_SYSTEM` incluye casos de UID -1.
- `CANCELLED`: incluye cancelaciones u otros marcados **no clasificables como no-show**, por lo que no asumir 100 % cancelaciones voluntarias.
- `DRAFT`: borrador estructural (tag 16, sujeto a prioridad de exclusiones).
- `RESCHEDULED_HISTORY`: copia histórica desplazada; no duplicar con cita vigente.
- `BLOCKED`: franjas bloqueadas; no son clientes con asistencia esperada.
- `IGNORED`: exclusión residual y regla histórica de clínica 8; documentar si se modifica.

**Nota técnica específica:** en `01_analytics_appointment_outcomes.sql`, `training_eligible` se inserta como `false` en la ETL inicial; el entrenamiento real **no depende de ese flag**, sino de los filtros de `14_appointment_training_dataset_v1.sql`. No asumir que `training_eligible=false` signifique 0 filas elegibles en el CSV.


---

## 4. Base de datos: magnitudes del origen y situación de la réplica

### 4.1 Conteos históricos observados en `raw_flow`

**Fuente:** `matriz_variables_tesis_flowww.md`, versión fechada el 7-10-2026, a partir de consultas anteriores. Son **conteos de la réplica examinada entonces**, no garantías de valores actuales ni certificación de igualdad en tiempo real con MySQL.

| Tabla | Filas observadas |
|---|---:|
| `raw_flow.clients` | 96.469 |
| `raw_flow.diary_gen` | 1.222.752 |
| `raw_flow.diary_det` | 3.037.406 |
| `raw_flow.laser_gen` | 302.539 |
| `raw_flow.laser_det` | 841.266 |
| `raw_flow.tickets_gen` | 574.951 |
| `raw_flow.tickets_det` | 1.594.618 |
| `raw_flow.vouchers` | 1.115.705 |

Comprobaciones informadas en esa auditoría temprana: 0 citas sin cliente enlazable para el criterio `DiaryGClientID > 0`; 302.382 citas con `DiaryGLaserGID > 0`; 0 sesiones enlazadas inexistentes; 35 citas con sesión y sin ticket enlazable; 11 con ticket abierto y 302.336 con ticket cerrado; 0 con ticket cerrado y `DiaryGNotAttended=-1` en ese subconjunto. La diferencia entre 302.336 tickets cerrados y 302.328 outcomes preliminares de atención se atribuyó a registros excluidos por prioridad de reglas.

**Clasificación preliminar anterior al ETL final** (no confundir con distribución del CSV ML):

| Grupo provisional | Filas |
|---|---:|
| `CANCELLED_OR_UNCLASSIFIED` | 514.075 |
| `EXCLUDED` | 339.651 |
| `ATTENDED` | 302.328 |
| `NO_SHOW_CANDIDATE` | 57.338 |
| `FUTURE` | 9.127 |
| `PAST_UNRESOLVED` | 187 |
| `SESSION_IN_PROGRESS` | 46 |
| **Total `diary_gen`** | **1.222.752** |

Son **estados preliminares excluyentes de esa consulta**, no el mismo campo/partición final de `analytics.appointment_outcomes`. El conteo de cancelaciones preliminares muestra por qué una marca masiva no es una etiqueta fiable de ausencia.

### 4.2 Estructuras analíticas principales

| Objeto | Función y estatus |
|---|---|
| `analytics.v_appointment_outcomes_v1` | Vista de clasificación base de FLOWww; reglas de exclusión, atención, candidatos no-show, cancelación, pendientes. |
| `analytics.appointment_outcomes` | Resultado tipado por cita; conserva identificadores, timestamps, marcas crudas, outcome, detail, tracking origin, flags de auditoría y causa de exclusión. |
| `analytics.appointment_events` | En v1 registra **eventos de `RESCHEDULED`** reconstruidos a partir de copias en `raw_flow.diary_gen`; su `event_at` se consideró candidato **no verificado** como instante exacto de reprogramación. No utilizarlo automáticamente como feature temporal. |
| `analytics.client_history_current` | Historial acumulado **actual**; útil para operaciones actuales, pero **no apto** para recrear historial de una reserva antigua. |
| `analytics.appointment_services` | Detalles con `service_phase` `SCHEDULED`, `PERFORMED`, `BILLED`; la carga revisada **solo puebla `SCHEDULED` desde `diary_det`**, no servicios realizados ni facturados. |
| `analytics.product_catalog_resolution` | Resolución de áreas/jerarquía del catálogo; advertencia de cambios históricos. |
| `analytics.ml_history_cumulative_v1` | Eventos atendidos/no-show con `known_at` y acumulados por cliente para unión temporal. |
| `analytics.appointment_training_dataset_v1` | Tabla final para exportación Python; una fila por cita elegible con 27 columnas y versión `BOOKING_BASELINE_V1`. |

**El repositorio Python no es dueño del ETL PostgreSQL:** `bds/analytics.sql` ofrece DDL/vistas y el contrato; los ETL históricos `01`, `03`, `07`, `13` y `14` se prepararon/ejecutaron por separado. No confundir `depi-ml export-dataset` con «construir desde cero la tabla de entrenamiento»: **export-dataset solo lee**.

### 4.3 Inventario cronológico de scripts SQL trabajados

| Script / artefacto | Función | Observación |
|---|---|---|
| `01_analytics_appointment_outcomes.sql` | ETL 01 de outcomes tipados, carga completa/UPSERT, tracking, exclusiones, `outcome_resolved_at`. | Requiere `v_appointment_outcomes_v1`; conserva UID -1. |
| `02_validacion_appointment_outcomes.sql` | Validación de ETL 01. | Diagnósticos, no modelo. |
| `03_analytics_appointment_events.sql` | ETL 02: copias de reprogramación y relación `DiaryGDelayedID`. | `event_at` no verificado para features. |
| `04_validacion_appointment_events.sql` | Validación de eventos. | Diagnósticos. |
| `05_analytics_client_history_current.sql` | Historial de cliente **a fecha actual**. | No usar para entrenamiento retrospectivo. |
| `06_validacion_client_history_current.sql` | Revisión de historial actual. | Diagnósticos. |
| `07_analytics_appointment_services.sql` | Servicios programados desde `diary_det` y catálogo. | No usar `laser_det` como input. Cuidado con bajas físicas en snapshots. |
| `08_validacion_appointment_services.sql` | Diagnósticos de servicios. | Verificar nulos/catálogo. |
| `13_ml_history_cumulative_v1.sql` | ETL 07 de historial acumulado conocido por cliente. | Ventana temporal con `known_at`; no usar saldo actual. |
| `14_appointment_training_dataset_v1.sql` | ETL 08 de dataset con 19 variables y filtros de elegibilidad. | Base oficial de la exportación. |
| `analytics_pr.sql`, `bds/raw_flow.sql`, `bds/analytics.sql` | Definiciones complementarias. | Verificar versión concreta antes de migraciones. |

Los nombres de scripts aquí corresponden a **archivos aportados durante la investigación**, no necesariamente a rutas de archivos clonables dentro de `depi-noshow-ml`; varias versiones de documentos o scripts compartidos existen. **No asumir que GitHub contiene la última copia de todos los ETL.** Preservarlos en un directorio versionado privado antes de modificar la base.

---

## 5. Definición exacta de la población de entrenamiento

### 5.1 Criterios implementados en `14_appointment_training_dataset_v1.sql`

En `base`, una cita entra inicialmente si:

1. `analytics.appointment_outcomes.outcome IN ('ATTENDED_COMPLETED','NO_SHOW')`.
2. Tiene `created_at_source` válido (usado como `prediction_at`).
3. La fecha de cita (`appointment_at`) es anterior a la fecha de construcción menos **1 día**, para evitar citas aún inmaduras; es un filtro temporal de ETL y debe revisarse si se regenera el dataset en otra fecha.
4. Se cumple `prediction_at < appointment_at`.
5. La anticipación entre creación y cita es ≤ **730 días**.
6. Existe cabecera `raw_flow.diary_gen` para la cita y cliente asociado en `raw_flow.clients`.
7. Existe al menos un agregado en `analytics.appointment_services` fase `SCHEDULED`.
8. **NO** existe evento `RESCHEDULED` de esa `appointment_id` en `analytics.appointment_events`.
9. El JOIN con historial es **opcional**; una persona sin historial previo entra con contadores `0` y recencias nulas.

**No incluye:** cancelaciones sin evidencia de no-show, borradores, bloques, copias históricas reprogramadas, futuros/pendientes, sesiones incompletas o registros sin detalle programado.

**Implicación científica:** el dataset representa **citas elegibles con resultado binario ya conocido según la reconstrucción**, no todas las reservas tomadas al momento de su creación. Aún puede existir **sesgo de selección** al excluir cancelaciones que se conocen después de reservar. Si en producción se puntúan todas las nuevas reservas, habrá una diferencia de población. La tasa global del dataset **NO** es la tasa global de inasistencia de toda la operación FLOWww.

### 5.2 Volumen consolidado del CSV real

| Clase | Etiqueta | Filas | Porcentaje del dataset |
|---|---:|---:|---:|
| Asistencia completada | `target=0` | 169.553 | ≈ 79,28 % |
| Candidato no-show | `target=1` | 44.320 | ≈ 20,72 % |
| **Total** | | **213.873** | **100 %** |

- `feature_version`: `BOOKING_BASELINE_V1`.
- Identificadores `appointment_id` y `client_id`: trazabilidad y auditoría, **no predictores**.
- Los valores nulos de features no se borraron; se tratan dentro del pipeline entrenado. Los validadores identificaron 0 duplicados de cita y 0 objetivos inválidos.
- SHA-256 original: `9241f2ee0a791ac4c0f8bc795e274338dd3e9b3bd216394ccd920a67bcaf0faa`.
- Documento de auditoría original: `methodology.md`; archivo inicial de ejemplo local: `data/exports/training_dataset_v1.csv`.

### 5.3 Las **27 columnas** del contrato CSV v1

Orden/roles canónicos en `src/depi_ml/datasets/schema.py`:

| # | Columna | Tipo lógico | Papel |
|---:|---|---|---|
| 1 | `appointment_id` | BIGINT | ID de cita; **no predictor**. |
| 2 | `client_id` | BIGINT | ID de cliente; **no predictor**. |
| 3 | `prediction_at` | TIMESTAMPTZ | Instante simulado de predicción; auditoría y cortes. |
| 4 | `appointment_at` | TIMESTAMPTZ | Fecha de cita; auditoría/coherencia, no input directo. |
| 5 | `target` | SMALLINT | Variable objetivo `0`/`1`. |
| 6 | `no_show_uid_minus_one` | BOOLEAN | Flag de auditoría/sensibilidad; no predictor. |
| 7 | `age_at_booking` | SMALLINT | **Predictor 1**. |
| 8 | `client_sex` | VARCHAR(1) | **Predictor 2**. |
| 9 | `booking_lead_days` | DOUBLE PRECISION | **Predictor 3**. |
| 10 | `appointment_month` | SMALLINT | **Predictor 4**. |
| 11 | `appointment_weekday` | SMALLINT | **Predictor 5**. |
| 12 | `appointment_hour` | SMALLINT | **Predictor 6**. |
| 13 | `duration_minutes` | INTEGER | **Predictor 7**. |
| 14 | `is_fwa` | BOOLEAN | **Predictor 8**. |
| 15 | `clinic_id` | INTEGER | **Predictor 9**. |
| 16 | `scheduled_service_lines` | INTEGER | **Predictor 10**. |
| 17 | `distinct_body_areas` | INTEGER | **Predictor 11**. |
| 18 | `single_body_area_id` | BIGINT | **Predictor 12**. |
| 19 | `has_medical_evaluation` | BOOLEAN | **Predictor 13**. |
| 20 | `has_type4_service` | BOOLEAN | **Predictor 14**. |
| 21 | `previous_attended` | INTEGER | **Predictor 15**. |
| 22 | `previous_no_show` | INTEGER | **Predictor 16**. |
| 23 | `previous_no_show_rate` | DOUBLE PRECISION | **Predictor 17**. |
| 24 | `days_since_previous_attended` | DOUBLE PRECISION | **Predictor 18**. |
| 25 | `days_since_previous_no_show` | DOUBLE PRECISION | **Predictor 19**. |
| 26 | `feature_version` | TEXT | Versión de feature engineering; no predictor. |
| 27 | `built_at` | TIMESTAMPTZ | Fecha de construcción de fila; no predictor. |

**Hay ocho columnas NO predictoras:** dos IDs, dos fechas, target, flag UID, versión y fecha de construcción. Los timestamps `label_recorded_at`, creados posteriormente en un archivo **auxiliar**, tampoco son predictores. No añadirlos a `X` bajo ninguna circunstancia.

### 5.4 Diccionario exhaustivo de las 19 variables predictoras

| Variable | Definición exacta o cálculo SQL v1 | Origen / riesgo relevante |
|---|---|---|
| `age_at_booking` | Edad en años completos a la fecha local de `prediction_at`, calculada con `ClientBirthDate`; edad admitida por SQL entre **16 y 95**, en caso contrario `NULL`. | `raw_flow.clients`; fecha de nacimiento puede haber sido corregida después. |
| `client_sex` | Código `ClientSex` de 1 carácter; SQL conserva solo `M`, `F`, `X`; otros → `NULL`. | `clients`; validar fiabilidad del valor histórico y efectos de trato diferencial. |
| `booking_lead_days` | `(appointment_at - prediction_at)` en segundos / 86.400. | Creación original `DiaryGCreated`; citación futura necesaria. |
| `appointment_month` | Mes local `1..12` de `appointment_at` en `America/Lima`. | Calendario. |
| `appointment_weekday` | `EXTRACT(ISODOW ...)`: lunes=1, …, domingo=7, calculado en Lima. | Contrato de código indica que convención requiere contraste; operación normalmente lunes–sábado. |
| `appointment_hour` | Hora local `0..23` de `appointment_at`. | Calendario. |
| `duration_minutes` | Diferencia entre hora fin/inicio de la cita programada, en minutos si inicio y fin son válidos; si no, `NULL`. | `diary_gen.DiaryGStart/End` del registro disponible; histórico as-of de duración no garantizado totalmente. |
| `is_fwa` | `DiaryGFWA <> 0`. | Indicador operativo FLOWww; su **semántica funcional exacta** sigue pendiente de acreditar. |
| `clinic_id` | ID de sede de `appointment_outcomes`. | Puede cambiar con desplazamientos; reprogramadas detectadas se excluyen. |
| `scheduled_service_lines` | `COUNT(*)` de filas en `appointment_services` con `service_phase='SCHEDULED'`. | De `diary_det`; no es cantidad real de zonas atendidas. |
| `distinct_body_areas` | Número `COUNT(DISTINCT body_area_product_id)` resuelto en `product_catalog_resolution`. | Jerarquía de catálogo actual; sensibilidad histórica. |
| `single_body_area_id` | Si hay exactamente **un área distinta**, su ID (`MIN`); si no, `NULL`. | Identificador categórico, **no cantidad ni orden**. |
| `has_medical_evaluation` | `BOOL_OR(is_medical_evaluation)` con default falso al agregar. | `appointment_services` y catálogo actual; interpretación/product code por validar. |
| `has_type4_service` | `BOOL_OR(product_type=4)` con default falso. | Tipo técnico del catálogo; no inferir que todo tipo 4 tenga significado clínico universal. |
| `previous_attended` | Asistencias conocidas del cliente antes de `prediction_at`, o `0` si no hay historial previo elegible. | `ml_history_cumulative_v1` con join `h.known_at < b.prediction_at`. |
| `previous_no_show` | Candidatos no-show anteriores conocidos, o `0`. | Mismo acumulado; incluye candidatos con UID -1 que existan en historia fuente. |
| `previous_no_show_rate` | `previous_no_show / (previous_attended + previous_no_show)` si denominador >0; de lo contrario `NULL`. | Ratio de resultados históricos observados, **no** de todas las citas/cancelaciones. |
| `days_since_previous_attended` | `(prediction_at - last_attended_at)` expresado en días; si no existe visita conocida, `NULL`. | Historia as-of. |
| `days_since_previous_no_show` | `(prediction_at - last_no_show_at)` en días; si no existe candidato previo, `NULL`. | Historia as-of. |

**No se entrenó con:** `DiaryGTags` finales de WhatsApp, saldo `VoucherAvailable` actual, `client_history_current`, resultado de `laser_det`, `tickets_det`, `target`, `client_id`, `appointment_id`, marca UID -1, `label_recorded_at`, fecha de construcción, ni catálogo de servicios efectivamente ejecutados.

### 5.5 Preprocesamiento ML actual

- El `FeatureContract` centraliza el orden de las 19 variables; transforma valores categóricos y valida la presencia de columnas.
- Se codifican como **categóricas**: `client_sex`, `clinic_id`, `single_body_area_id`, `is_fwa`, `has_medical_evaluation`, `has_type4_service`, `appointment_month`, `appointment_weekday` y `appointment_hour`.
- Otras variables se pasan como numéricas; se aplican transformaciones mediante un `ColumnTransformer`/pipeline scikit-learn, imputación y one-hot para que el modelo admita nulos/categorías ausentes.
- El ajuste de imputer/encoder se realiza **solo con train**, y luego se reutiliza idéntico para validación e inferencia. Categorías nunca vistas se toleran. Se preserva la dimensión incluso si una columna numérica está totalmente nula.
- `class_weight='balanced_subsample'` en RF y `scale_pos_weight = negativos/positivos` en XGBoost original; ambos pesos se calculan desde **entrenamiento**, no desde validación/prueba. Estos pesos pueden alterar calibración de probabilidades.
- El pipeline y contrato compartido para la API Python están en `training/preprocessing.py` y `datasets/schema.py`.

---

## 6. Cómo se reconstruye el historial y por qué importa el tiempo

### 6.1 `analytics.ml_history_cumulative_v1`

Archivo original: `13_ml_history_cumulative_v1.sql`.

- Evento 0 «visita atendida»: `appointment_outcomes.outcome='ATTENDED_COMPLETED'`, ticket cerrado y **`raw_flow.laser_gen.LaserGCloseDate AT TIME ZONE 'America/Lima'`** como `known_at`.
- Evento 1 «candidato no-show»: `outcome='NO_SHOW'` y **`outcome_resolved_at`** como `known_at`.
- Solo incluye eventos con `client_id > 0`, timestamp observado y `known_at <= CURRENT_TIMESTAMP` a la construcción del ETL.
- Acumula por cliente, ordenando por `known_at, appointment_id`: atenciones/no-shows vistos **hasta incluir el evento de esa fila** y últimas marcas conocidas.
- Se indexa por `(client_id, known_at DESC, appointment_id DESC)` para buscar la historia anterior a una nueva reserva.

En `14_appointment_training_dataset_v1.sql` se utiliza conceptualmente:

```sql
LEFT JOIN LATERAL (
  SELECT h.prior_including_event_attended,
         h.prior_including_event_no_show,
         h.last_attended_at, h.last_no_show_at
  FROM analytics.ml_history_cumulative_v1 h
  WHERE h.client_id = b.client_id
    AND h.known_at < b.prediction_at
  ORDER BY h.known_at DESC, h.appointment_id DESC
  LIMIT 1
) h ON TRUE
```

Al tomar un acumulado que incluye el último evento **anterior** a la reserva, no se incluye por la lógica SQL la etiqueta de la propia cita cuando se resuelve posteriormente. **No afirmar ausencia absoluta de todas las fugas**, pues otros atributos, disponibilidad real y modificaciones tardías requieren revisión.

### 6.2 Auditoría SQL del historial: resultados importantes

Se consultó la relación entre `known_at` y `appointment_at` para `ml_history_cumulative_v1`:

| Evento | Filas en historial consultado | `known_at < appointment_at` | `known_at > appointment_at + 7 días` |
|---|---:|---:|---:|
| Asistencia (`event_kind=0`) | 302.325 | 7.442 | 18 |
| Candidato no-show (`event_kind=1`) | 57.338 | 0 | 0 |

La fecha máxima tardía observada en **historial completo** fue del orden de 220 días. **Estos conteos son del historial más amplio y no deben confundirse con los 213.873 ejemplos finales.**

Se investigaron específicamente las **7.442 asistencias con `known_at < appointment_at`**:

- **3.410** de esas filas estaban en el dataset de entrenamiento.
- **0** de esas 3.410 tenían `known_at < prediction_at`.
- Las **3.410** tenían marca de cierre **después de crear la reserva y antes de la hora programada**.

**Conclusión estrecha:** las marcas tempranas del subconjunto de entrenamiento **no eran anteriores a la creación de la cita**. Se interpretaron como advertencias operativas, no como bloqueadores automáticos. No se identificó la causa individual (pruebas, error de especialista, atención anticipada u otro motivo).

### 6.3 Historial comparable y límites de la auditoría del CSV

El reporte `methodology.md` observó:

- `history_comparable_client_events`: **210.432**.
- `history_count_exceeds_observed_csv_events`: **148.448**.

La segunda cifra **NO prueba por sí sola fuga de información**, porque el historial construido con `appointment_outcomes` abarca eventos que **no entraron** a la muestra elegible del CSV (p. ej. por modificaciones, límites, disponibilidad, filtros). Es una señal de que se requiere evidencia externa del join y la historia, no una razón para borrar automáticamente 148.448 filas.

---

## 7. Disponibilidad de etiquetas y excepciones de cierres manuales

### 7.1 Problema original y decisión aplicada

Inicialmente se propuso una cota general `appointment_at + label_delay_hours`. Se descartó porque **hay cierres administrativos tardíos**, incluso meses después; no era defendible decir que todas las asistencias se conocían 24 horas después.

Se añadió un **archivo auxiliar por cita**, unido por `appointment_id`, sin cambiar las 27 columnas originales:

```text
appointment_id,target,outcome,label_recorded_at,label_source
```

Definición:

- `target=0`: `LaserGCloseDate AT TIME ZONE 'America/Lima'` (cierre manual de atención).
- `target=1`: `appointment_outcomes.outcome_resolved_at` (derivado de tracking FLOWww).

`label_recorded_at` es **marca registrada / proxy de disponibilidad**, NO prueba definitiva de que el sistema/usuario conociera ese resultado a esa hora. Cambios retroactivos e ingestión más tardía siguen siendo posibilidades metodológicas.

### 7.2 Integridad verificada del auxiliar

Reporte real `label_availability_report.json` de PROMPT4:

| Comprobación | Resultado |
|---|---:|
| Filas auxiliares | **213.873** |
| IDs coinciden con CSV | Sí |
| Targets/outcomes coinciden | Sí |
| IDs duplicados | **0** |
| Etiquetas faltantes | **0** |
| Outcomes inválidos | **0** |
| Fechas posteriores a instantánea origen | **0** |
| Fechas temporales **críticas** | **0** |
| Asistencias cerradas entre reserva y cita (advertencias) | **3.410** |
| Cierres de asistencias >24 horas tras cita | **38** (auditoría SQL previa) |
| Cierres de asistencias >7 días tras cita | **13** (auditoría SQL previa) |
| Máxima demora de asistencia | **220,854444… días** |
| Máxima demora de no-show | **0,124965… días**, unas 3 h |
| Estado técnico | `validated_with_operational_warnings` |
| Estado metodológico | `pending_human_review` |

- No-show: 44.320 etiquetas, ninguna anterior a cita, ninguna faltante.
- Asistencia: 169.553 etiquetas, 3.410 registradas antes de **hora de cita** pero **después de hora de reserva**.
- La interpretación de negocio confirmó que las especialistas **cierran manualmente**; posibles errores/pruebas son **hipótesis**, no diagnósticos de esas 3.410 filas.

**Cambios de PROMPT4:**

- **Asistencia crítica:** `label_recorded_at <= prediction_at`.
- **Asistencia advertencia:** `prediction_at < label_recorded_at < appointment_at`.
- **No-show crítico:** `label_recorded_at < appointment_at`.
- **Cualquier etiqueta:** `label_recorded_at > source_snapshot_at` crítica.
- Advertencias se conservan y participan con los cortes normales; no se trasladan artificialmente al horario de cita, ni se descartan automáticamente.

### 7.3 SHA-256 y artefactos relacionados

| Artefacto | Identidad |
|---|---|
| CSV original `training_dataset_v1.csv` | `9241f2ee0a791ac4c0f8bc795e274338dd3e9b3bd216394ccd920a67bcaf0faa` |
| Auxiliar `training_dataset_v1_labels.csv` | `df6b9a58cdf5a65b6b808a776100768c565ab402aa5512b42a4d5ed0e48292ae` |
| Manifest de auxiliar | `9a3d45fb72093b10cbdf6641d48e89122baaccbce6161db773c2b53b67e2a292` |
| Versión de regla temporal | `outcome_aware_v2` |

El hash del **manifest CSV original** debe leerse de su archivo si se necesita una reproducción exacta; no inventarlo. PROMPT4 reportó que los seis artefactos protegidos (CSV/manifest/reporte tanto original como auxiliar) mantuvieron sus hashes, aunque el **reporte de calidad del auxiliar originalmente etiquetaba 3.410 como críticas** y la revalidación posterior las reclasificó. La revalidación recalcula reglas desde datos en vez de confiar en el estado antiguo grabado en manifest.

---

## 8. Auditorías de outcomes, cancelaciones, borradores y UID -1

### 8.1 Dataset de análisis por mes **de cita** (citas binarias elegibles)

Se calculó la frecuencia de candidatos no-show por mes sobre el CSV de entrenamiento, no sobre toda FLOWww:

| Mes de **cita** | Citas elegibles | Candidatos no-show | Asistencias | Tasa no-show |
|---|---:|---:|---:|---:|
| **Junio 2026** | 13.111 | 2.743 | 10.368 | **20,92 %** |
| **Julio 2026** | 10.703 | 765 | 9.938 | **7,15 %** |
| **Agosto 2026** | 11.552 | 552 | 11.000 | **4,78 %** |
| **Septiembre 2026** | 11.690 | 302 | 11.388 | **2,58 %** |

**Precauciones:** son citas que satisfacen la definición de población elegible y tienen resultado ya clasificado. Las cifras de `appointment_outcomes` son mayores porque incluyen citas excluidas del dataset. Mes de **cita** no equivale a mes de **reserva**, utilizado en división temporal/validación. La caída coincide aproximadamente con un cambio de negocio, pero **no constituye una inferencia causal ni prueba formal de concept drift**.

Otras alertas identificadas por `methodology.md`: abril 2025 pasó de 14,21 % a 21,07 % (5.438 citas), octubre 2025 de 18,53 % a 24,32 % (11.424), junio 2026 bajó desde 27,59 % a 20,92 % (13.111), julio 2026 bajó de 20,92 % a 7,15 % (10.703). Los meses inicial/final del dataset pueden ser parciales.

### 8.2 Estados por mes en **todos** los outcomes de `analytics.appointment_outcomes`

Consulta compartida por el usuario (archivo `Result_18.json`): junio–septiembre de 2026 por `appointment_at` en Lima.

| `outcome` | Junio | Julio | Agosto | Septiembre |
|---|---:|---:|---:|---:|
| `ATTENDED_COMPLETED` | 20.482 | 22.706 | 23.290 | 23.450 |
| `NO_SHOW` | 3.689 | 1.538 | 969 | 575 |
| `CANCELLED` | 25.015 | 17.872 | 28.749 | **69.811** |
| `DRAFT` | 4.180 | 9.818 | 13.787 | 15.880 |
| `RESCHEDULED_HISTORY` | 23.566 | 27.174 | 23.468 | 22.201 |
| `BLOCKED` | 533 | 830 | 949 | 921 |
| `IN_PROGRESS` | 7 | 1 | 4 | 1 |
| `UNKNOWN` | 2 | 4 | 2 | 3 |
| `IGNORED` | 0 | 0 | 0 | 1 |

**Hallazgo:** el descenso en `NO_SHOW` también aparece en outcomes, no solo en el CSV elegible; las asistencias permanecen relativamente estables. **Al mismo tiempo**, septiembre presenta 69.811 registros `CANCELLED`; no interpretar automáticamente esa cifra como cancelaciones humanas reales ni inferir una tasa causal de ausencia por esos volúmenes.

### 8.3 Descomposición de `CANCELLED` por origen de tracking

Se agrupó por mes, `classification_detail` y `tracking_origin`. **Todas** las cancelaciones analizadas tenían `classification_detail='CANCELLED_OR_UNCLASSIFIED'`:

| Mes | Origen `UID_MINUS_ONE` | Origen `USER_POSITIVE` | Total `CANCELLED` |
|---|---:|---:|---:|
| Junio 2026 | **22.995** | 2.020 | 25.015 |
| Julio 2026 | **15.560** | 2.312 | 17.872 |
| Agosto 2026 | **26.974** | 1.775 | 28.749 |
| Septiembre 2026 | **68.852** | 959 | 69.811 |

**Conclusión cauta:** el gran volumen de septiembre se concentra en marcados con `UID_MINUS_ONE` y outcome no distinguible entre cancelación y otra condición. No demuestra que todas fueran cancelaciones reales ni que el sistema esté trasladando las inasistencias a `CANCELLED` de manera concreta. Es un caso importante para la validez de etiquetas y la interpretación del cambio comercial.

### 8.4 Borradores y marcas de cancelación / tickets

Consulta SQL de solo lectura sobre outcomes (`DRAFT`) de junio a septiembre:

| Mes | `DRAFT` | `DRAFT` con marca de cancelación | `DRAFT` con ticket cerrado | `CANCELLED` fuera de `DRAFT` |
|---|---:|---:|---:|---:|
| Junio | 4.180 | 4.178 | **2** | 25.015 |
| Julio | 9.818 | 9.818 | **0** | 17.872 |
| Agosto | 13.787 | 13.787 | **0** | 28.749 |
| Septiembre | 15.880 | 15.877 | **3** | 69.811 |
| **Total** | **43.665** | **43.660** | **5** | **141.447** |

Interpretación:

- `DRAFT` y `CANCELLED` son outcomes finales **mutuamente excluyentes** por las prioridades del ETL.
- Un `DRAFT` **sí puede tener marca cruda de cancelación**; de hecho casi todos los examinados la tenían. Eso no lo convierte en `CANCELLED` porque el filtro estructural de borrador gana prioridad.
- Solo **5 borradores** de ese período tenían ticket cerrado; para el negocio son anomalías por proceso/error, no hechos que requieran reconstruir toda la muestra. Están excluidos de ML.
- La coincidencia de ticket cerrado en `DRAFT` no prueba por sí sola qué servicio se atendió. No usar esa cifra para estimar ausencias reales.
- Se decidió **documentar y seguir**, sin eliminar datos ni alterar los ETL por estas 5 excepciones.

### 8.5 Auditoría de candidatos no-show marcados con UID -1

Datos previamente compartidos (`uid_groups.csv`, `uid_month.csv`):

| Cohorte del CSV ML | Total |
|---|---:|
| Candidatos no-show totales | 44.320 |
| Candidatos no-show con `no_show_uid_minus_one=true` | **1.557** (≈ 3,51 % de candidatos) |
| Candidatos no-show con UID distinto de -1 | **42.763** |
| Asistencias | **169.553** |
| Dataset si se excluyen solo candidatos no-show UID -1 | **212.316** filas antes de purga temporal |

Por mes de cita, el número de candidatos UID -1 en el conjunto elegible fue aproximadamente: junio **88**, julio **56**, agosto **76**, septiembre **89**. Su proporción entre los no-shows de esos meses fue cerca de **3,21 %**, **7,32 %**, **13,77 %** y **29,47 %**, respectivamente.

**Interpretación:** la caída total de candidatos no-show **no es explicada solo por UID -1**. La proporción relativa UID -1 aumenta porque el total de candidatos disminuye fuertemente. El experimento `without_uid_minus_one` excluye **esas filas candidatas** de las particiones pero **no recalcula** todos los historiales de clientes que pudieron contar eventos UID -1; por tanto es una sensibilidad parcial, no una «etiqueta purificada».

### 8.6 Auditoría por clínica

El archivo `clinic_month.csv` presentaba **206 filas clínica-mes**, con 15 IDs de clínicas en el conjunto revisado (`1,2,3,4,5,6,7,9,10,11,12,13,15,16,17`). **Las 12 clínicas comparables en junio y julio mostraron una disminución de la tasa de candidatos no-show**. Ejemplos citados en revisión:

- Clínica 1: junio ~20,72 % → julio ~5,24 %.
- Clínica 2: junio ~19,93 % → julio ~5,72 %.
- Clínica 5: junio ~27,51 % → julio ~13,68 %.

Las sumas por fila `rows = no_show + attended` fueron coherentes; las agregaciones mensuales coinciden con la tabla de la §8.1. Este patrón transversal apoya que el cambio no se localiza únicamente en una clínica. **No descarta** cambios en mezcla de clientes, reglas del sistema, cobertura, registros de cancelación o política comercial.

### 8.7 Cambio comercial de junio–julio 2026 [NEGOCIO]

- **Antes:** en general era posible hacer una reserva sin pago anticipado y pagar al acudir.
- **Después:** como regla general se exige **comprar previamente un bono/voucher antes de reservar**.
- **Excepción:** persona que acude presencialmente el mismo día puede ser atendida si hay disponibilidad.
- **Fecha:** **aproximada junio–julio 2026**, día exacto desconocido. No etiquetar arbitrariamente «15 de junio», «1 de julio» u otra fecha fija como evento probado.
- **Hipótesis razonable:** mayor compromiso de pago anticipado podría reducir inasistencias; **no se identificó causalidad exclusiva**. También pueden operar recategorizaciones, cambios de seguimiento, diferencias de cartera, atención presencial, madu­rez de etiquetas o variación por sede.
- **Consecuencia para tesis:** evidencia de posible cambio de régimen/distribución. Debe evaluarse capacidad de generalización del modelo entrenado con patrones anteriores. El cambio **no se incluyó** como predictor en PROMPT6, ni se alteraron cortes de partición.

---

## 9. Validación técnica del CSV y controles observados

### 9.1 Auditoría inicial (`methodology.md`)

El informe local documentó **0 errores técnicos críticos** entre sus verificaciones observables de estructura y coherencia. Entre los controles que resultaron en `0`:

- IDs faltantes/duplicados, targets inválidos, ausencia de fechas, reservas con `prediction_at >= appointment_at`, flags UID en asistencias;
- `booking_lead_days` negativo; duración negativa; número de líneas o áreas negativo; conteos de historial negativos;
- tasas de historial incompatibles con numerador/denominador; recencias sin historia correspondiente o historia previa con recencia faltante;
- desacuerdo entre mes, hora o día de semana calculado y fecha de cita según regla local usada;
- edad fuera de rango `0..120`, duración igual a cero, áreas superiores a líneas o ID de área única incoherente.

**Interpretación correcta:** estas pruebas verifican consistencia y contrato; **no certifican** automáticamente que las features fueran conocidas en el instante de reservar, ni que la etiqueta no-show sea perfecta. Por eso se separa **«control técnico pasado»** de **«revisión metodológica acreditada»**.

### 9.2 Ocho controles metodológicos obligatorios

| Control | Qué hay avanzado | Qué sigue sin acreditarse plenamente |
|---|---|---|
| `history_point_in_time` | ETL v1 utiliza acumulados con `h.known_at < prediction_at`; investigación de 3.410 cierres tempranos; sin evidencia de etiqueta propia antes de reservar en ese grupo. | Auditoría de disponibilidad real, completitud/eventos fuente y cambios históricos del historial. |
| `booking_snapshot` | Confirmada lógica de copia/reprogramación y exclusión de **162.905** casos detectados; `laser_det`/`tickets_det` no se usan como predictores. | Garantía as-of para horario, duración, sede, sexo y detalle programado en todos los casos no detectados; cambios sin traza. |
| `label_availability` | Archivo auxiliar con 0 errores críticos, 3.410 advertencias y cierres tardíos tratados por fecha registrada. | Probar disponibilidad **real** del resultado, posibles cierres retroactivos, ingestiones tardías y sesgo de purga. |
| `catalog_snapshot` | Experimento `without_catalog` desarrollado. | Versionado de catálogo por fecha o prueba de que IDs/características eran invariantes históricamente. |
| `label_definition` | Reglas y diferencias de `NO_SHOW`, `CANCELLED`, UID -1, exclusiones analizadas. | Confirmación de verdad terreno y origen de registros automatizados, concordancia con negocio. |
| `selection_bias` | Se conoce el alcance de citas binarias y exclusión de cancelaciones. | Estimar cómo generaliza una clasificación condicionada a citas que posteriormente tuvieron outcome binario. |
| `uncertain_semantics` | `appointment_weekday` en SQL es `ISODOW`; sensibilidad sin `is_fwa` realizada. | Definición operativa verificable de `is_fwa` y pruebas de semántica en la interfaz/sistema. |
| `temporal_label_stability` | Caída por mes de cita y clínicas; cambio comercial documentado; métricas mensuales de validación. | Confirmar régimen posterior con prueba final y monitoreo prospectivo; separar explicación causal de correlación. |

**Estado real del archivo** `methodology_review_template.json` entregado: 8 controles con `status='pending'`, `evidence=''`, `reviewer=''`, `reviewed_at=''`. Los hashes del dataset y del auxiliar están vinculados. El modo normal de `train` bloquea hasta **revisión real y documentada**; el modo explícito `--exploratory` permite desarrollo y comparación preliminar **sin fingir aprobación metodológica**.

### 9.3 Matiz importante sobre `without_catalog`

En `training/preprocessing.py`:

```python
CATALOG_COLUMNS = {
    'distinct_body_areas',
    'single_body_area_id',
    'has_medical_evaluation',
    'has_type4_service',
}
```

Por ello **`without_catalog` NO elimina `scheduled_service_lines`**, que cuenta filas de servicios programados; quita las cuatro características ligadas a resolución del catálogo/tipo/área. Se llegó a considerar eliminar también las cinco variables de servicio por temor a ediciones sin traza, pero la aclaración del negocio sobre reprogramaciones/copia llevó a **no imponer ese cambio**. Una comparación «sin todo servicio programado» requeriría un **experimento nuevo**, no confundirlo con el `without_catalog` ya evaluado.

---

## 10. Separación temporal de datos y prevención de fuga (*leakage*)

### 10.1 Cortes candidatos usados efectivamente

Los cortes usados en los experimentos de Fase 2, con reloj Lima explícito, son:

```text
validation_start = 2026-04-01T00:00:00-05:00
      test_start = 2026-07-01T00:00:00-05:00
        test_end = 2026-10-01T00:00:00-05:00
```

**La partición se asigna por `prediction_at` (momento de reserva), NO por `appointment_at` ni aleatoriamente.**

Reglas de maduración temporal:

- **Train:** `prediction_at < validation_start` y `label_recorded_at < validation_start`.
- **Validación:** `validation_start <= prediction_at < test_start` y `label_recorded_at < test_start`.
- **Prueba:** `test_start <= prediction_at < test_end` y `label_recorded_at < exported_at` del dataset original.
- El corte es estricto `<`, nunca `<=` en la admisión de etiquetas.
- No se imputan fechas ausentes ni se supone un retardo global fijo.
- Los datos de 2026-07 a 2026-09 son **holdout de prueba**, NO utilizados en entrenamiento exploratorio ni en la optimización.

### 10.2 Vista previa real de particiones obtenida en PROMPT4

| Grupo por fecha de reserva | Reservas en ventana | Retenidas con etiqueta madura | No-shows retenidos | Etiquetas excluidas por corte | Advertencias operativas retenidas |
|---|---:|---:|---:|---:|---:|
| **Entrenamiento** | 135.767 | **130.042** | **30.502** | **5.725** | **1.983** |
| **Validación** | 45.639 | **41.427** | **9.924** | **4.212** | **599** |
| **Prueba (solo vista previa, no evaluada)** | 30.917 | **30.917** | **1.076** | **0** | **660** |
| Fuera de ventanas de reserva (no particionado) | **1.550** | — | — | — | — |

La vista previa de PROMPT4 muestra la población potencial de prueba porque valida datos, pero en PROMPT5/PROMPT6 **`include_test=False`**, así que **no se materializó la prueba para modelado ni se calcularon métricas sobre ella**.

La vista previa también indicó:

- Entrenamiento: 3.514 asistencias y 2.211 no-shows excluidos por `label_recorded_at` posterior/igual al corte.
- Validación: 3.629 asistencias y 583 no-shows excluidos por el mismo criterio.
- Sin fechas de etiqueta faltantes.
- Advertencias temporales **en ventana**: train 2.042, validación 665, prueba 660; **retenidas**: train 1.983, validación 599, prueba 660.
- Intersección de clientes entre train y validación: **16.965**; entre train y prueba: **10.727** (solo conteos de la vista previa). **No se diseñó** un estudio de clientes nunca vistos; son clientes nuevos y recurrentes con fechas de reserva distintas.
- Prevalencia aproximada: train **23,45 %**, validación **23,96 %**, prueba prevista **3,48 %**. Esta brecha refuerza la preocupación por cambio de régimen; NO es una evaluación de rendimiento en prueba.

### 10.3 Sesgos/limitaciones de la estrategia

1. Los cortes son **por reserva**, mientras la cohorte mensual de ETL del §8.1 es **por cita**. No comparar los porcentajes como si fuera el mismo denominador.
2. Excluir etiquetas todavía no conocidas al corte evita una forma de fuga, pero puede **seleccionar diferencialmente** resultados o casos según rapidez de registro; registrar exclusiones por clase es indispensable.
3. La validación abril–junio incluye junio, posible transición comercial; la prueba julio–septiembre sería el examen real fuera del régimen anterior.
4. `label_recorded_at` puede no equivaler a visibilidad efectiva del evento en el sistema. Si no hay evidencia de ingestión y ediciones, la partición es un **aproximado operacional**.
5. El uso repetido de validación para comparar configuraciones genera optimismo de selección. **No realizar una ronda indefinida de tuning buscando llegar a AUC 0,80**.
6. El servicio de inferencia futuro debe garantizar que todas las variables procedan de un instante concreto «as-of», no de los valores actuales de otra fecha.

---


## 11. Experimentos de Fase 2: diseño y métricas

### 11.1 Experimento exploratorio inicial — PROMPT5 [REPORTADO CODEX]

**Fecha/contexto:** 8-10-2026. `depi-ml train --exploratory`, semilla `42`, umbral de clasificación **diagnóstico** `0,5`; 12 modelos = tres métodos (`baseline`, Random Forest, XGBoost) × cuatro variantes. Estado de cada salida **`EXPLORATORY_NOT_VALIDATED`**.

| Variante | Variables / muestra |
|---|---|
| `full` | Las **19** predictoras del contrato v1. |
| `without_is_fwa` | Retira únicamente `is_fwa`. |
| `without_catalog` | Retira **cuatro** features de área/tipo/evaluación (`CATALOG_COLUMNS`), **no** el conteo `scheduled_service_lines`. |
| `without_uid_minus_one` | Mantiene las 19 features, pero **excluye filas** de candidatos no-show con flag UID -1; no vuelve a construir los historiales acumulados. |

**Filas efectivas:**

- `full`, `without_is_fwa` y `without_catalog`: entrenamiento **130.042**, validación **41.427**.
- `without_uid_minus_one`: entrenamiento **129.390**, validación **40.824**.
- Baseline: `DummyClassifier(strategy='prior')`, probabilidad constante igual a prevalencia en entrenamiento. Como predice una probabilidad inferior a 0,5, a umbral 0,5 **no marca ningún positivo**: recall, precision y F1 igual a `0` por convención, **sin que eso invalide su rol de referencia probabilística**.

### 11.2 Tabla **completa** de los 12 modelos iniciales

Estas son **métricas de validación**, no prueba. PR-AUC se calcula como **Average Precision (AP)**, no área trapezoidal. **Brier menor es mejor**; las demás métricas, mayor es mejor (con matiz de balance y costos).

| Experimento | Modelo | ROC-AUC | AP | Recall | Precision | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` | **Baseline** | 0,5000 | 0,2396 | 0,0000 | 0,0000 | 0,0000 | 0,1822 |
| `full` | **Random Forest** | **0,7502** | **0,5126** | **0,5091** | 0,4822 | **0,4952** | 0,1756 |
| `full` | **XGBoost** | 0,7247 | 0,4731 | 0,4177 | 0,4847 | 0,4487 | **0,1717** |
| `without_is_fwa` | Baseline | 0,5000 | 0,2396 | 0,0000 | 0,0000 | 0,0000 | 0,1822 |
| `without_is_fwa` | Random Forest | 0,7484 | 0,5087 | 0,5217 | 0,4728 | 0,4960 | 0,1787 |
| `without_is_fwa` | XGBoost | 0,7211 | 0,4638 | 0,4064 | 0,4803 | 0,4403 | 0,1725 |
| `without_catalog` | Baseline | 0,5000 | 0,2396 | 0,0000 | 0,0000 | 0,0000 | 0,1822 |
| `without_catalog` | Random Forest | **0,7503** | **0,5167** | 0,4412 | **0,5172** | 0,4762 | **0,1656** |
| `without_catalog` | XGBoost | 0,7190 | 0,4628 | 0,3686 | 0,4917 | 0,4213 | 0,1694 |
| `without_uid_minus_one` | Baseline | 0,5000 | 0,2283 | 0,0000 | 0,0000 | 0,0000 | 0,1762 |
| `without_uid_minus_one` | Random Forest | **0,7603** | 0,5145 | 0,5273 | 0,4766 | 0,5007 | 0,1721 |
| `without_uid_minus_one` | XGBoost | 0,7292 | 0,4676 | 0,4227 | 0,4777 | 0,4485 | 0,1682 |

**Interpretación técnica:**

- Random Forest `full` domina a XGBoost `full` en ROC-AUC, AP, recall y F1 con umbral fijo 0,5. XGBoost tiene **Brier menor** y una precision ligeramente superior. No hay un ganador universal de todas las métricas.
- RF `without_catalog` conserva ROC-AUC ~0,75, mejora levemente AP (0,5167) y reduce Brier (0,1656) frente a RF completo, pero **reduce recall** (0,4412 vs 0,5091). Su relevancia es metodológica y operativa: menos dependencia de atributos históricos del catálogo.
- RF `without_uid_minus_one` presenta ROC-AUC 0,7603, el mayor ROC-AUC de esta tabla, **pero utiliza una población diferente**; comparar siempre con baseline y prevalencia de ese subconjunto, no interpretarlo como aumento puro por arquitectura.
- La presencia de AP notablemente por encima del baseline muestra discriminación útil en la cohorte de validación. **No demuestra despliegue productivo, calibración, eficacia de recordatorios ni mejora de negocio.**
- En términos del usuario: el `0,49` visto antes correspondía al **F1-score**, no al ROC-AUC. ROC-AUC de RF original es ~0,75 y de XGBoost original ~0,72.

### 11.3 Hiperparámetros iniciales efectivamente usados (código PROMPT5)

```python
RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=2,
)

XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    scale_pos_weight=negativos_entrenamiento / positivos_entrenamiento,
    random_state=42,
    n_jobs=2,
)
```

- `baseline` mantiene la probabilidad base aprendida del train; no es modelo operativo de alertas.
- 19 variables en modelo completo; one-hot y nulos se manejan en pipeline.
- El umbral `0,5` fue **fijo, diagnóstico**, sin elección de bandas de riesgo ni optimización de F1 en prueba.

---

## 12. Optimización acotada — PROMPT6 [REPORTADO CODEX]

### 12.1 Propósito, alcance y salvaguardas

Se hizo **una única ronda fija** para evitar optimización interminable:

- Dos experimentos: `full` y `without_catalog`.
- Por experimento: **3 variantes nuevas RF** + **5 variantes nuevas XGBoost** = **8 candidatos nuevos**.
- En ambos experimentos: **16 modelos nuevos**, más referencias originales (2 RF, 2 XGB, 2 baselines) reevaluadas sin reentrenar.
- Semilla `42`; umbral fijo `0,5`; mismas ventanas y disponibilidad de etiquetas; sin búsqueda adaptativa ni early stopping; prohibida evaluación de prueba.
- El código reproducible se encuentra en `src/depi_ml/training/optimization.py`.

| Nombre de configuración | Cambios frente al original |
|---|---|
| `rf_leaf20` | `min_samples_leaf=20` |
| `rf_features03` | `max_features=0.3` |
| `rf_regularized` | `min_samples_leaf=20`, `max_features=0.3`, `max_depth=16` |
| `xgb_shallow` | `n_estimators=300`, `max_depth=3`, `min_child_weight=5`, `reg_lambda=5` |
| `xgb_depth5` | `n_estimators=300`, `max_depth=5`, `min_child_weight=5`, `reg_lambda=5` |
| `xgb_deep_regularized` | `n_estimators=400`, profundidad 6, `learning_rate=0.03`, `min_child_weight=10`, `reg_lambda=10`, `gamma=0.1` |
| `xgb_slow` | `n_estimators=400`, `learning_rate=0.03`, `min_child_weight=5`, `reg_lambda=5` |
| `xgb_unweighted` | `n_estimators=300`, `min_child_weight=5`, `reg_lambda=5`, `scale_pos_weight=1` |

El código declaró anticipadamente un filtro **descriptivo**, no estadístico: ganancia AP absoluta ≥ 0,01, caída ROC-AUC ≤ 0,002, empeoramiento Brier ≤ 0,002 y caída AP por mes ≤ 0,01. Cumplirlo **no** equivale a validación operativa ni selección definitiva.

### 12.2 Resultados principales compartidos por Codex

La siguiente tabla reúne originales y **el candidato nuevo con mayor AP** por algoritmo/variante, no los 16 candidatos completos. Consultar `optimization_report.md` para todos los candidatos en el entorno local.

| Experimento | Modelo/configuración | ROC-AUC | AP | Recall | Precision | F1 | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| `full` / `without_catalog` | Baseline original | 0,5000 | 0,2396 | 0,0000 | 0,0000 | 0,0000 | 0,1822 |
| `full` | RF original | **0,7502** | **0,5126** | 0,5091 | 0,4822 | 0,4952 | 0,1756 |
| `full` | RF `rf_regularized` | 0,7364 | 0,5117 | 0,4959 | 0,4783 | 0,4869 | 0,1745 |
| `full` | XGB original | 0,7247 | 0,4731 | 0,4177 | 0,4847 | 0,4487 | 0,1717 |
| `full` | XGB `xgb_unweighted` | 0,7278 | **0,4832** | **0,0833** | 0,7299 | 0,1496 | 0,1659 |
| `without_catalog` | RF original | **0,7503** | **0,5167** | 0,4412 | 0,5172 | 0,4762 | 0,1656 |
| `without_catalog` | RF `rf_leaf20` | 0,7491 | 0,5090 | **0,6050** | 0,4277 | 0,5011 | 0,1858 |
| `without_catalog` | XGB original | 0,7190 | 0,4628 | 0,3686 | 0,4917 | 0,4213 | 0,1694 |
| `without_catalog` | XGB `xgb_unweighted` | 0,7272 | **0,4815** | **0,0659** | 0,7667 | 0,1214 | 0,1681 |

**Corrección de lectura:** la fila de «Baseline original» del reporte se refiere al mismo baseline en los dos experimentos de población idéntica; no confundirla con el baseline distinto de `without_uid_minus_one` (AP 0,2283).

### 12.3 Qué aprendimos realmente de PROMPT6

1. **Ningún candidato RF nuevo superó la AP del RF original** en sus experimentos respectivos. No se justifica otra ronda de tuning solo por pequeñas fluctuaciones.
2. XGBoost `xgb_unweighted` mejoró AP **+0,0101** en `full` y **+0,0187** en `without_catalog`, pero **recall cayó drásticamente** a 8,33 % y 6,59 %, respectivamente, con umbral fijo 0,5. Aumentó precisión entre los pocos marcados positivos, no ganó capacidad equilibrada de detectar todos los no-shows.
3. `rf_leaf20` sin catálogo elevó recall a **60,50 %** con menor precision y Brier mayor: otro perfil de trade-off, no superior en AP. No adoptarlo como decisión operativa sin costos/umbral y validación externa.
4. Se conservaron todos los originales y los nuevos sin sobreescribir. **No se seleccionó/promovió modelo definitivo.**
5. Las probabilidades no están calibradas para expresar niveles de riesgo operativos. Brier es diagnóstico; un buen Brier global puede convivir con mala calibración en subgrupos o meses.
6. Reutilizar validación para elegir configuraciones puede sesgar al alza los resultados. La prueba apartada debe seguir apartada.

### 12.4 Métricas **mensuales por mes de reserva**, experimento completo

| Mes de **reserva** | N retenidas de validación | Prevalencia no-show | AP RF original | AP XGB original | AP XGB `unweighted` |
|---|---:|---:|---:|---:|---:|
| Abril 2026 | **16.357** | **26,92 %** | **0,5433** | 0,5047 | 0,5114 |
| Mayo 2026 | **17.451** | **25,59 %** | **0,5357** | 0,4952 | 0,5052 |
| Junio 2026 | **7.619** | **13,85 %** | **0,3584** | 0,3154 | 0,3366 |

**Ojo con bases de fecha/población:** junio **20,92 %** de la auditoría principal = **mes de cita y población completa elegible**; junio **13,85 %** de PROMPT6 = **mes de reserva y filas retenidas tras purga de etiquetas**. No existe contradicción: son denominadores y relojes diferentes.

La AP de junio cae de forma relevante, pero parte de esa disminución se relaciona con la **prevalencia menor**; la referencia trivial de AP también disminuye cuando hay menos positivos. Codex reportó ROC-AUC de RF original cercano a 0,75 en los meses. Sin una prueba posterior al cambio, **no afirmar desempeño de julio a septiembre**.

### 12.5 Interpretación y decisión provisional

- **Candidatos técnicos provisionales para avanzar con FastAPI:** RF `full` original y RF `without_catalog` original. El primero detecta más no-shows a umbral fijo 0,5; el segundo tiene mejor AP/Brier en esa validación y menos dependencia de catálogo actual. **No hay ganador científico final**.
- **XGBoost** permanece como comparador central exigido por tesis, con ROC-AUC aproximado 0,72–0,73 y AP 0,46–0,48 en los experimentos principales. No ocultar métricas desfavorables de recall de variantes no ponderadas.
- **No seguir buscando infinitamente AUC 0,80:** el objetivo contractual es **AUC ≥ 0,80 sobre prueba**, y aún no se puede afirmar éxito; desarrollar arquitectura y prototipo es compatible con mejorar el modelo después.

---

## 13. Glosario de métricas: qué significan y errores comunes

| Métrica | Qué mide | Lectura adecuada aquí |
|---|---|---|
| **ROC-AUC** | Probabilidad de que un positivo escogido al azar obtenga score mayor que un negativo escogido al azar (con convención de empates). | Referencia aleatoria ≈ **0,50**. **RF 0,7502 no significa 75,02 % de citas acertadas**. Meta PI1: **≥0,80 en prueba**. |
| **AP / PR-AUC** | Precisión promediada a través de niveles de recall; especialmente informativa con minoría positiva. | Baseline completo ≈ **0,2396**, RF completo ≈ **0,5126** en validación. Cambia con prevalencia. |
| **Recall / sensibilidad** | `TP / (TP + FN)`: proporción de no-shows reales detectados con umbral elegido. | RF completo **0,5091** a 0,5: ~51 de cada 100 positivos en validación. |
| **Precision** | `TP / (TP + FP)`: fracción de alertas positivas que fueron verdaderos no-shows. | RF completo **0,4822** a 0,5: ~48 de cada 100 alertas fueron correctas. |
| **F1** | Media armónica de precision y recall, dependiente de umbral. | RF completo **0,4952**. **No confundir con ROC-AUC**. |
| **Brier** | Promedio del error cuadrático de probabilidad frente a objetivo 0/1; **menor = mejor**. | RF completo **0,1756**, XGB completo **0,1717**. No acredita por sí solo calibración local y futura. |
| **Matriz de confusión** | `[[TN,FP],[FN,TP]]` a umbral fijo. | Ayuda a estimar carga de llamadas/alertas, falsos positivos y ausencias omitidas. |
| **Accuracy** | `(TP+TN)/N`; puede ser alta simplemente por clase mayoritaria. | **No debe dominar** el criterio con baja prevalencia de no-show. |

- La selección de un umbral operacional será una decisión **coste/capacidad/valor de riesgo**, no simplemente «0,5 porque así entrenamos».
- AUC y AP utilizan scores, y no mejoran por cambiar solo el umbral de clasificación.
- Para anunciar probabilidades «riesgo 60 %» debe comprobarse **calibración** y estabilidad tras el cambio de reglas comerciales.
- Las curvas ROC/PR, histogramas de probabilidad, matrices de confusión, calibración, importancias por permutación y SHAP son **diagnósticos**, no demostración causal de por qué no asiste una persona.

---

## 14. Cronología de trabajo, decisiones y commits

| Período / hito | Trabajo realizado | Resultado / decisión |
|---|---|---|
| Septiembre–principios octubre 2026 | PI1, estado del arte, documentación de FLOWww y variables candidatas. | Arquitectura, problema y criterios redactados; MySQL→PostgreSQL definido. |
| Principios de octubre | Migración/replicación inicial de FLOWww; SQL de `appointment_outcomes`, eventos, servicios, historia. | Copia en `raw_flow`, reconstrucción en `analytics`; cautela con cancelación y no-show. |
| 7–8 octubre | `13_ml_history_cumulative_v1.sql`, `14_appointment_training_dataset_v1.sql`, exportación CSV. | 213.873 filas, 27 columnas; ETL historia as-of. |
| 8 octubre, `5fc7059` | Inicialización de `jossrc/depi-noshow-ml`. | Repositorio Python, estructura base. |
| 8 octubre, `af41bc2` | Fase 2 inicial. | Auditoría/EDA, split y experimentos, revisión estricta. |
| Auditorías reales | UID -1, clínicas, cambio de tasas, outcomes, borradores, cancelaciones, cierres manuales y reprogramaciones. | 5 DRAFT con ticket cerrado; 0 reprogramaciones detectadas dentro del train; cambio comercial aproximado junio–julio. |
| `16f36aa` — PROMPT3 | Archivo auxiliar de `label_recorded_at` y nuevos splits. | Primera validación marcó 3.410 cierres anteriores a hora de cita como críticos. |
| `acebfaf` — PROMPT4 | Corrección de política temporal por resultado. | 0 anomalías temporales críticas, 3.410 advertencias; 128 pruebas aprobadas según Codex. |
| `047851a` — PROMPT5 | Modo `train --exploratory`. | 12 modelos entrenados, test aislado; **148 tests aprobados** según Codex. |
| `77ea0fe` — PROMPT6 | Ronda acotada `optimize-exploratory`. | 16 candidatos nuevos; análisis mensual; 168 tests aprobados según Codex; no modelo final. |
| **8 octubre 2026, esta memoria** | Consolidación para retomar APIs y continuar hasta 2027. | Conservación de contexto, cifras, fuentes, límites y decisiones. |

**Importante:** los identificadores de commit completos deben consultarse con `git log`; la abreviatura aquí sirve como ayuda de navegación. La última referencia comprobada fue `77ea0fe6f8422c152412bda863aef6b8dbabab1b`.

---

## 15. Repositorio `depi-noshow-ml`: mapa técnico y estado real

### 15.1 Estructura versionada comprobada al corte

```text
depi-noshow-ml/
├── README.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example              # plantilla; NO subir .env real
├── .gitignore
├── bds/
│   ├── raw_flow.sql
│   └── analytics.sql
├── docs/
│   ├── dataset_contract.md
│   ├── guia_local.md
│   └── phase2.md
├── data/exports/.gitkeep    # datos y reportes exportados son locales
├── src/depi_ml/
│   ├── cli.py
│   ├── config.py
│   ├── db/connection.py
│   ├── datasets/
│   │   ├── schema.py
│   │   ├── validator.py
│   │   ├── exporter.py
│   │   ├── label_exporter.py
│   │   └── label_availability.py
│   ├── analysis/
│   │   ├── audit.py
│   │   ├── dataset.py
│   │   ├── label_report.py
│   │   └── reports.py
│   ├── training/
│   │   ├── preprocessing.py
│   │   ├── splits.py
│   │   ├── review.py
│   │   ├── experiments.py
│   │   └── optimization.py
│   └── evaluation/
│       ├── metrics.py
│       ├── reports.py
│       └── runner.py
└── tests/
    ├── test_dataset_exporter.py
    ├── test_dataset_validator.py
    ├── test_label_availability.py
    ├── test_exploratory.py
    ├── test_optimization.py
    ├── test_phase2.py
    └── ...
```

**Todavía NO aparece una implementación de FastAPI** en el árbol fuente examinado; no inventar `api.py` o endpoints existentes. El plan arquitectónico es integrarlo en el **mismo repositorio Python**, reutilizando `datasets/schema.py` y `training/preprocessing.py`.

### 15.2 Comandos CLI disponibles según código

| Comando `depi-ml` | Función | Conexión / precaución |
|---|---|---|
| `inspect-dataset` | Consulta estructura, estadísticas e integridad del dataset analítico. | PostgreSQL, **READ ONLY**. |
| `export-dataset` | Genera CSV, manifest, reporte de calidad con comparación snapshot DB. | PostgreSQL, **READ ONLY**. No volver a ejecutar por rutina. |
| `analyze-dataset` | EDA/auditoría local sobre CSV + manifest. | Sin PostgreSQL. |
| `export-label-availability` | Extrae `label_recorded_at` por cita, vinculado a CSV/manifest originales. | PostgreSQL, **READ ONLY**, no sobrescribir archivos existentes. |
| `validate-label-availability` | Valida auxiliar y previsualiza cortes, reporta advertencias/críticos. | Sin PostgreSQL, sin entrenamiento. |
| `train` | Entrena solo si existe revisión metodológica acreditada. | Rechaza controles pendientes. |
| `train --exploratory` | Entrena/valida sin firmar controles metodológicos, con integridad obligatoria. | Solo train/validation; `EXPLORATORY_NOT_VALIDATED`; test vedado. |
| `optimize-exploratory` | Una ronda fija en `full` y `without_catalog`, compara con referencia. | Solo train/validation; test vedado. |
| `evaluate` | Evaluación de prueba de un experimento con revisión metodológica aceptada. | **Rechaza** experimentos exploratorios. No ejecutar sobre PROMPT5/PROMPT6. |

### 15.3 Comandos seguros de consulta y pruebas en entorno local

Desde la raíz del repo, con Python 3.12+ y entorno virtual existente:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,ml,explain]'
python -m pytest -q

depi-ml --help

# Para validar el auxiliar existente sin acceder a PostgreSQL,
# utiliza el comando real documentado en el bloque siguiente.
```

Comando del validador local:

```bash
depi-ml validate-label-availability \
  --csv data/exports/training_dataset_v1.csv \
  --labels data/exports/training_dataset_v1_labels.csv \
  --validation-start 2026-04-01T00:00:00-05:00 \
  --test-start 2026-07-01T00:00:00-05:00 \
  --test-end 2026-10-01T00:00:00-05:00 \
  --output reports/label-review-nuevo-nombre
```

**Cuidado:** si ya existe la carpeta indicada en `--output`, el comando se niega a sobreescribirla. No repetir exportación ni entrenamiento solo para verificar una métrica; reutilizar reportes y hashes existentes.

### 15.4 Artefactos locales de las fases

```text
data/exports/
  training_dataset_v1.csv
  training_dataset_v1_manifest.json
  training_dataset_v1_quality_report.json
  training_dataset_v1_labels.csv
  training_dataset_v1_labels_manifest.json
  training_dataset_v1_labels_quality_report.json

reports/
  phase2-review-01/
    methodology.md, audit.json, clases y gráficos EDA, ...
  phase2-labels-prompt4-20261008/
    label_availability_report.json
    methodology_review_template.json

experiments/
  phase2-exploratory-prompt5-20261008/
    experiment.json
    audit.json
    review.json
    validation_comparison.{md,csv,json}
    <variante>/<modelo>/model.joblib
    <variante>/<modelo>/validation/...

  phase2-optimization-prompt6-20261008/
    plan.json
    experiment.json
    findings.md
    optimization_report.md
    validation_comparison.{csv,json}
    validation_monthly.{csv,json}
    monthly_<experimento>_<modelo>.png
    modelos y hashes de candidatos ...
```

**Los paths anteriores describen ubicaciones reportadas/convenciones locales; no significa que sus binarios estén disponibles en GitHub.** `data/exports/`, `reports/` y `experiments/` están ignorados en Git para proteger datos y artefactos potencialmente sensibles. Para retomar en otra máquina hay que conservar una **copia privada de esos artefactos locales**, verificar hashes y ejecutar los comandos con el entorno compatible; el código solo de GitHub no restaura modelos entrenados ni el CSV automáticamente.

### 15.5 Pruebas y reproducibilidad

- Codex informó **128 tests aprobados tras PROMPT4**, **148 tras PROMPT5** y **168 tras PROMPT6**; esto es **resultado reportado de ejecución local**, no test repetido en este documento.
- PROMPT5 verificó hashes de 12 modelos y conservación de 6 archivos originales. PROMPT6 dijo conservar **177 archivos protegidos** y no acceder a PostgreSQL ni a prueba.
- Para reproducir se necesitan: commit del código, Python/dependencias, 27 columnas CSV + manifest, auxiliar de etiquetas + manifest, JSON de revisión, semilla, cortes, hiperparámetros y salidas originales, todos con hashes cuando corresponda.
- Cambiar versión del pipeline, de bibliotecas, de dataset o del auxiliar puede invalidar comparabilidad de artefactos `.joblib`. **No cargar modelos de procedencia desconocida**: deserializar joblib ejecuta código Python potencialmente peligroso.

---


## 16. Desarrollo siguiente: FastAPI, integración y APIs

> Esta sección es un **plan técnico sugerido para el siguiente desarrollo**. **NO** describe endpoints ya implementados. Las decisiones finas de contrato, persistencia, autenticación e infraestructura deben validarse antes de producción.

### 16.1 Por qué el siguiente paso es FastAPI

El usuario indicó limitación de tiempo y decidió **continuar construyendo la aplicación** sin seguir optimizando AUC indefinidamente. Esta decisión es coherente con la arquitectura modular de la tesis siempre que:

- Se etiquete el modelo actual como **provisional/exploratorio**.
- El API reutilice **exactamente** las 19 features y el pipeline entrenado, sin entrenar de nuevo con datos en una request.
- No se habiliten por omisión scores o recomendaciones automáticas productivas sobre clientes reales sin validación previa.
- La comparación RF/XGBoost, la documentación de limitaciones y la prueba final permanezcan pendientes y trazables.

### 16.2 Interfaz lógica recomendada del microservicio Python

Propuesta mínima de endpoints (nombres y estructura a decidir en código):

| Propuesta | Método | Uso | Precaución |
|---|---|---|---|
| `/health` | GET | Disponibilidad de proceso Python. | «Proceso activo» no garantiza que el modelo esté cargado. |
| `/ready` | GET | Modelo/carga/pipeline compatibles y listos. | No publicar rutas internas o secretos. |
| `/v1/predict` | POST | Recibir features validadas, devolver probabilidad de `target=1` y versión de modelo. | Acceso interno/autenticado; no consultar datos sensibles sin necesidad. |
| `/v1/model-info` | GET interno | Contrato (features, versión, hash, semilla, estado metodológico). | Evitar filtrar rutas físicas, manifest sensible o información de clientes. |

**Propuesta de cuerpo de predicción** (valores ficticios de demostración, no cita real ni contrato ya aprobado):

```json
{
  "age_at_booking": 31,
  "client_sex": "F",
  "booking_lead_days": 4.5,
  "appointment_month": 11,
  "appointment_weekday": 3,
  "appointment_hour": 15,
  "duration_minutes": 30,
  "is_fwa": false,
  "clinic_id": 1,
  "scheduled_service_lines": 1,
  "distinct_body_areas": 1,
  "single_body_area_id": 1001,
  "has_medical_evaluation": false,
  "has_type4_service": false,
  "previous_attended": 3,
  "previous_no_show": 1,
  "previous_no_show_rate": 0.25,
  "days_since_previous_attended": 40.0,
  "days_since_previous_no_show": 180.0
}
```

> `single_body_area_id=1001`, clínica y demás números de este ejemplo son **inventados únicamente para mostrar formato**. No usar este JSON para validar semántica de servicios reales. No introducir `target`, `outcome`, `appointment_id`, `client_id`, `label_recorded_at`, `no_show_uid_minus_one` ni fechas de resultado como características.

**Ejemplo de respuesta conceptual** (también ficticia):

```json
{
  "no_show_probability": 0.37,
  "model_family": "random_forest",
  "model_version": "PROVISIONAL_EXPLORATORY",
  "methodology_status": "EXPLORATORY_NOT_VALIDATED",
  "risk_band": null
}
```

No declarar automáticamente «riesgo medio» para 0,37 sin calibración, umbrales/capacidad definidos y aprobación del negocio. La inferencia puede devolver `risk_band: null` y una probabilidad claramente etiquetada como experimental en prototipo.

### 16.3 Flujo del backend Node.js/Fastify

1. Recibir una solicitud autorizada referente a la **reserva actual**.
2. Leer únicamente atributos de la cita disponibles **al instante en que se generará el score**; adjuntar `prediction_timestamp` para trazabilidad.
3. Construir `age_at_booking`, anticipación, calendario, servicios programados, sede e historial con **reglas temporales equivalentes** a las de la tabla de entrenamiento. No tomar «último resultado conocido hoy» si la predicción se pretende evaluar retrospectivamente.
4. Validar nullabilidad y tipos según `datasets/schema.py` + implementación de FastAPI. Si la fase `without_catalog` es la elegida, seleccionar explícitamente el contrato de 15 variables efectivas, sin cambiar silenciosamente un artefacto de 19.
5. Invocar FastAPI dentro de red protegida, con timeout, validación de respuesta y trazabilidad del modelo/versiones.
6. Devolver a Next.js probabilidad, estado experimental y explicaciones permitidas; persistir como score histórico solo si se diseña una tabla con hash/fecha/versión/consentimiento apropiados.
7. Nunca permitir que el navegador cargue `.joblib` ni credenciales PostgreSQL/MySQL; la inferencia se realiza en servidor.

**Aclaración clave sobre cálculos as-of:** aunque en el ML offline existe `analytics.ml_history_cumulative_v1`, para puntuar nuevas citas debe resolverse cómo actualizar/consultar el historial **sin utilizar resultados futuros** y cómo manejar sincronización por lotes. El backend no puede simplemente enviar `client_id` a Python esperando que Python lea el CSV histórico: eso provocaría acoplamiento y posibles predicciones incorrectas para clientes/servicios nuevos.

### 16.4 Propuesta de organización interna FastAPI

```text
src/depi_ml/
  api/                       # PROPUESTA, no existe todavía
    app.py                   # inicialización FastAPI
    routes.py                # predict/health/readiness
    schemas.py               # validación Pydantic de contrato
    service.py               # cargar y llamar pipeline una sola vez
    model_registry.py        # versión y verificación de metadatos
  training/preprocessing.py  # YA EXISTE: reutilizar
  datasets/schema.py         # YA EXISTE: fuente única de columnas
```

Decisiones que deberán tomarse explícitamente: si cargar en `startup` un modelo `joblib` local, cómo aprobar/checkear hash y versión, dónde ubicar artefactos en un contenedor, cómo autenticar llamadas Fastify→FastAPI, cuánto tarda el score, cómo instrumentar logs sin datos sensibles y cómo facilitar reemplazo posterior de Random Forest por XGBoost u otra versión.

### 16.5 Validaciones y pruebas mínimas deseadas antes de integrar con Next.js

- `GET /health` y `/ready` con comportamiento correcto ante modelo no disponible/corrupto.
- Pruebas de request con los 19 campos (o contrato de sensibilidad seleccionado), nulos admitidos, tipos erróneos, extra fields, categorías desconocidas, rangos/imposibilidades y operaciones sobre fechas.
- **Paridad offline/API:** para una muestra sintética controlada, `predict_proba` de FastAPI igual al del pipeline guardado; verificar nombres, orden, tipos y nulabilidad.
- Hash y metadata del modelo; solo archivos propios y confiables, nunca binarios externos.
- Registro sin `client_id` en logs públicos y sin contraseña; control de acceso y límites de peticiones.
- Respuesta coherente ante timeouts y modelo no disponible; no devolver un score inventado.
- Medición de latencia para contrastar meta PI1 de API promedio **≤ 5 segundos**, sin confundirla con meta predictiva.
- Manejo del error y rollback de versión del modelo (promoción manual, no automática por la mejor métrica exploratoria).
- Cuando exista integración completa, pruebas end-to-end Node→Python→UI y juicio de expertos sobre comprensión/uso de alertas.

### 16.6 Plan de fases técnicas pendiente hasta 2027

| Fase siguiente | Entregable útil | Estado |
|---|---|---|
| **FastAPI prototipo** | Health, readiness, predict con RF provisional, carga segura de modelo, validación 19 features, pruebas unitarias. | **Pendiente**; próximo frente de desarrollo. |
| **Cálculo online de features** | Servicio Node/Fastify de construcción as-of por reserva + integridad histórica. | Pendiente. |
| **Integración API** | Fastify autentica/orquesta FastAPI, protocolo JSON y errores. | Pendiente. |
| **Frontend Next.js** | Pantallas de agenda, consulta y score en modo demostración. | Pendiente. |
| **Aprobación metodológica** | Evidencia para 8 controles, revisión humana firmada, hash y código congelados. | Pendiente; no falsear verificaciones. |
| **Evaluación final** | Desbloquear `evaluate` únicamente para experimento acreditado; reportar julio–septiembre de 2026 y generalización. | Pendiente, prueba intacta. |
| **Calibración / gestión de riesgo** | Calibración de probabilidades, umbrales/bandas sustentados, costos y capacidad de Call Center. | Pendiente. |
| **Validación operativa** | Expertos, usabilidad, pruebas funcionales/latencia y eventual piloto. | Pendiente. |
| **Actualización de datos en producción** | Pipeline CSV incremental/snapshot, borrados, versionado, drift, alertas, backup, governance. | Pendiente. |
| **PI2 / redacción final** | Resultados reproducibles, discusión de no cumplimiento/cumplimiento de AUC 0,80, limitaciones y evidencias. | Pendiente. |

---

## 17. Qué NO se debe afirmar ni repetir sin evidencia

1. **«ROC-AUC 75 % = 75 % de citas acertadas»**: falso. ROC-AUC no es accuracy.
2. **«Ya cumplimos AUC ≥ 0,80 de PI1»**: falso; validación principal ~0,75 y prueba aún no evaluada.
3. **«El mejor modelo ya fue seleccionado»**: no; RF completo y sin catálogo son candidatos provisionales. XGB se mantiene como comparador.
4. **«0 errores críticos = no existe leakage»**: no; algunas fuentes retrospectivas y snapshots aún requieren investigación.
5. **«Las 3.410 citas son definitivamente pruebas o errores de especialistas»**: la causa específica no está probada. Sí se verificó que su cierre ocurrió después de crear reserva.
6. **«Cada CANCELLED representa cliente que canceló»**: no; clase `CANCELLED_OR_UNCLASSIFIED` y marcado masivo UID -1 impiden esa certeza.
7. **«El pago anticipado causó toda la caída en no-shows»**: solo coincidencia temporal consistente con la hipótesis, no causalidad exclusiva.
8. **«La prueba tiene AUC X»**: no hay métrica final de julio–septiembre en los experimentos revisados.
9. **«El experimento sin UID -1 supera al completo de modo directamente comparable»**: usan poblaciones diferentes y no se recalcula toda la historia.
10. **«`without_catalog` elimina todos los servicios»**: falso; conserva `scheduled_service_lines`.
11. **«FastAPI y la aplicación ya están operativos»**: falso según último árbol comprobado; son siguientes etapas.
12. **«El repo público contiene los CSV reales o los `.joblib`»**: archivos locales ignorados; conservar backups privados.
13. **«El registro de resultado se conoció exactamente en `label_recorded_at`»**: es un proxy, podría haber modificaciones/ingesta.
14. **«El modelo reducirá costos/ausencias X %»**: no medido; sería necesario experimento operativo.
15. **«Se puede bajar el criterio de éxito retroactivamente para decir que fue válido»**: cualquier modificación exige sustento formal, versión y aprobación, no uso oportunista de prueba.

---

## 18. Preguntas abiertas y datos que conviene obtener posteriormente (no necesarios para crear esta memoria)

**No bloquear el desarrollo de FastAPI por todos estos puntos**, pero no perderlos:

1. **Fecha exacta del cambio comercial**: ¿hay circular, correo o registro de fecha de obligatoriedad del bono y posibles sedes con rollout distinto? No inventar.
2. **Prueba temporal final**: ¿se puede acreditar suficientemente la revisión metodológica para evaluar una sola vez la cohorte julio–septiembre? Evitar descubrir el AUC final antes de fijar protocolo.
3. **Snapshot de reserva**: ¿existen logs, bitácora o reglas que demuestren cambios de sede, duración, fecha, sexo y servicios en citas no detectadas como `RESCHEDULED`?
4. **Catálogo histórico**: ¿`ProductType`, `ProductEvaluation`, zonas y padres cambian con el tiempo? ¿Existe timestamp/versión fiable?
5. **`is_fwa`**: definición exacta, valores esperados, cuándo se asigna, si es conocido a la reserva.
6. **Candidatos UID -1**: ¿por qué tantos `CANCELLED_OR_UNCLASSIFIED` provienen del sistema, especialmente septiembre? ¿existen operaciones batch documentadas?
7. **Etiquetas**: ¿hay timestamps de creación/edición/ingesta separados de `LaserGCloseDate` / `tracking_at` para demostrar disponibilidad real?
8. **Reprogramaciones por edición**: confirmar sistemáticamente con trazas de un caso de servicio modificado si siempre se generan copia/evento detectable.
9. **Detección de reservas especiales**: walk-ins, reservas con bonos ya adquiridos, cambios de canal y condiciones de elegibilidad en producción.
10. **Impacto del filtrado de cancelaciones**: comparar población puntuable online con población binaria entrenada; diseñar evaluación consistente.
11. **Privacidad y permisos**: definir responsables, retención, finalidad y control de acceso a predicciones y datos personales.
12. **Modelos y artefactos**: respaldar privado `experiment.json`, `.joblib`, hashes y gráficos; Git no los contiene.
13. **Validación de sistema**: obtener criterios de expertos y plan de usabilidad/latencia, seguridad, operación y despliegue.
14. **Evaluación probabilística**: calibración, utilidad del score tras cambio comercial y elección de bandas/acciones.

**Datos NO requeridos ahora:** no hace falta pedir nuevamente los resultados de las consultas de junio–septiembre, UID, borradores, las 3.410 fechas ni las 162.905 reprogramaciones; quedan registrados en esta memoria.

---

## 19. Handoff operativo: cómo retomar el proyecto meses después

### 19.1 Checklist corto para una nueva conversación o Codex

1. **Entregar esta memoria** y aclarar fecha de corte (8-10-2026); cualquier cambio posterior tiene precedencia y debe anexarse con fecha/commit.
2. Abrir el repositorio: <https://github.com/jossrc/depi-noshow-ml> y consultar el HEAD actual. Comparar contra `77ea0fe6f8422c152412bda863aef6b8dbabab1b` antes de presumir archivos disponibles.
3. Leer `README.md`, `docs/dataset_contract.md`, `docs/phase2.md`, `src/depi_ml/datasets/schema.py`, `src/depi_ml/training/preprocessing.py`, `src/depi_ml/training/experiments.py`, `src/depi_ml/training/optimization.py`, `src/depi_ml/training/review.py`.
4. Confirmar que el proyecto local tiene CSV y auxiliar exactos (hashes del §7.3), manifiestos y modelos confiables. **Si solo hay código GitHub, faltan artefactos reales**; no inventar métricas ni modelos.
5. Respetar el estado `EXPLORATORY_NOT_VALIDATED`; los ocho checks siguen `pending` salvo que una revisión real posterior muestre lo contrario.
6. Si el objetivo de la siguiente sesión es desarrollo: empezar por **FastAPI** reutilizando pipeline y contrato; no ejecutar nuevamente SQL ni optimización sin motivo.
7. Si el objetivo es investigación: revisar resultados de pruebas anteriores y evidencia metodológica; solo luego plantear congelar modelo/umbral, evaluar prueba y discutir AUC ≥0,80.
8. Si el objetivo es desplegar: exigir controles de privacidad, paridad de features, auth, latencia, logging, versionado, monitoreo, calibración y validación de negocio.

### 19.2 Mensaje de contexto reutilizable para copiar en un futuro chat

> Estamos continuando el proyecto TESIS 2026–2027: sistema web predictivo de no-shows en empresa de estética, con Next.js/React + Node.js/Fastify + Python/FastAPI + PostgreSQL. La fase Python en `jossrc/depi-noshow-ml` se revisó hasta el commit `77ea0fe` de 8-10-2026. La BD fuente es FLOWww MySQL 5 replicada a PostgreSQL 18.4 (`raw_flow`, `analytics`). El dataset `appointment_training_dataset_v1` tiene 213.873 filas, 27 columnas y 19 features, hash `9241f2...`; las etiquetas son candidatos de no-show. La validación usa reserva cronológica: train 130.042, validation 41.427, holdout previsto 30.917. Ya existen resultados exploratorios: RF full ROC-AUC 0,7502/AP 0,5126/F1 0,4952; XGB full ROC-AUC 0,7247/AP 0,4731/F1 0,4487; RF without_catalog AUC 0,7503/AP 0,5167. No alcanzamos aún la meta PI1 AUC ≥0,80 en prueba; prueba NO evaluada y metodología con 8 checks pendientes. Cambio comercial aproximado junio–julio 2026: bono prepago obligatorio salvo walk-ins del mismo día. PROMPT5 entrenó 12 modelos; PROMPT6 comparó 16 candidatos y no seleccionó modelo definitivo. Las 3.410 marcas de asistencia anteriores al horario de cita son advertencias operativas, no fechas anteriores a la creación de reserva; sin errores temporales críticos. El próximo paso de desarrollo es FastAPI con RF provisional y el mismo pipeline de preprocessing. No repetir auditorías resueltas, trabajar paso a paso, distinguir hecho vs hipótesis y no promover el modelo a producción sin validación.

### 19.3 Procedimiento para actualizar esta memoria durante 2027

- **No editar silenciosamente conclusiones antiguas.** Añadir una sección «Actualización YYYY-MM-DD», responsable/contexto, commit, artefactos, hashes nuevos, decisión, evidencia y limitaciones modificadas.
- Si se cambia dataset o features, aumentar versión de contrato (`BOOKING_BASELINE_V1` → siguiente solo cuando se apruebe) y documentar ruptura de comparabilidad.
- Si se introduce nueva etiqueta/no-show verdad terreno, documentar definición y explicar cómo afecta comparaciones anteriores.
- Separar resultados de **validación**, **prueba**, **piloto productivo** y **métricas de negocio**; nunca mezclarlos en una tabla sin identificar población y reloj.
- Anotar explicitamente si el nuevo modelo está `exploratory`, `reviewed`, `test_evaluated`, `calibrated`, `pilot` o `production_approved` y por quién.
- Registrar cada migración de BD: lote, tablas afectadas, función inversa/backup, riesgos de borrado y comandos (sin secretos).
- Mantener los archivos reales de entrenamiento/validación y ETL en respaldo privado versionado; enlazar desde este índice sin insertar filas personales.

---

## 20. Fuentes primarias y trazabilidad documental

### 20.1 Material del proyecto consultado para este documento

1. **`PI1 Trabajo Investigación v4.docx`** (3-10-2026): portada, resumen, problema, objetivos, indicadores IND-04, benchmarking, arquitectura, validación, anexos de calidad/riesgos. Especialmente: sección 1.2.2 «Indicadores de Éxito»; «Línea Base de Calidad» (`AUC ≥ 0,80` en prueba); anexo de riesgos (`AUC < 0,70` como ejemplo).
2. **`documentacion_bd_flow_tesis.md`** (documentación viva de FLOWww) y **`matriz_variables_tesis_flowww.md`** (7-10-2026): esquema, fuentes, datos replicados, WhatsApp, reglas de negocio e inventario de variables candidatas.
3. **`bd_flow_05102026.sql`** y **`function.ts`**: DDL MySQL y código operativo original usado durante análisis de interpretación (no ejecutados aquí para producir esta memoria).
4. SQL del usuario: `01_analytics_appointment_outcomes.sql`, `03_analytics_appointment_events.sql`, `07_analytics_appointment_services.sql`, `13_ml_history_cumulative_v1.sql`, `14_appointment_training_dataset_v1.sql`, y validadores complementarios `02`, `04`, `06`, `08`.
5. **`methodology.md`**: reporte de coherencia del CSV 213.873; alertas/controles y revisión pendiente.
6. **`uid_groups.csv`, `uid_month.csv`, `clinic_month.csv`**: auditorías UID -1 y disminución por clínica/mes (datos agregados; no filas de clientes).
7. **`Result_18.json`** y capturas de las consultas **Result 19–21**: outcomes mensuales, descomposición de cancelaciones, borradores/tickets y disponibilidad de etiquetas.
8. **`training_dataset_v1_labels_quality_report.json`**, **`label_availability_report.json`** y **`methodology_review_template.json`**: 0 errores críticos, 3.410 advertencias y particiones; checks pendientes.
9. **Resúmenes de ejecución Codex** compartidos en la conversación para **PROMPT4**, **PROMPT5** y **PROMPT6**: pruebas aprobadas, modelos, métricas y resultados de optimización. **No son auditoría independiente de los binarios**.
10. **Repositorios y documentación GitHub** examinados con conector GitHub al cierre de la memoria:
    - <https://github.com/jossrc/depi-noshow-ml>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/docs/dataset_contract.md>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/docs/phase2.md>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/datasets/schema.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/training/splits.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/training/preprocessing.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/training/experiments.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/training/optimization.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/training/review.py>
    - <https://github.com/jossrc/depi-noshow-ml/blob/main/src/depi_ml/datasets/label_availability.py>

### 20.2 Información no recuperada y no inventada

- No se inspeccionó un `model.joblib` real ni se reentrenó en la máquina del asistente; los indicadores PROMPT5/6 se citan **tal como fueron reportados por Codex**.
- `optimization_report.md`, `findings.md`, curvas, SHAP y modelos son artefactos locales **no presentes en el repo público** por `.gitignore`, aunque sus rutas y conclusiones fueron reportadas.
- No existe evidencia de fecha exacta de la nueva política de bono ni validación causal.
- No se dispone de una revisión metodológica firmada con ocho checks verificados; no simularla.
- No se ha consultado la base PostgreSQL de producción para esta memoria ni se ha alterado ninguna tabla.
- No se garantizó que el estado de las tablas/código permanezca igual después del commit de corte; futuras sesiones deben volver a verificar la versión actual del repositorio.

---

## 21. Cierre de situación al 8 de octubre de 2026

**Lo logrado:** recuperación e interpretación de FLOWww; SQL analítico; dataset ML de 213.873 reservas elegibles; contrato de 19 predictoras; ETL de historial as-of; controles de integridad; auditorías de estados, cancelaciones, borradores, UID -1, clínicas, reprogramaciones y cierres manuales; manejo por cita de `label_recorded_at`; train/validation temporales; cuatro sensibilidades; 12 modelos exploratorios; optimización acotada de 16 candidatos; métricas e interpretación mensual; repositorio Python versionado.

**Resultado predictivo principal:** RF completo ROC-AUC **0,7502**, AP **0,5126**, recall **0,5091**, F1 **0,4952**; XGBoost completo ROC-AUC **0,7247**, AP **0,4731**, recall **0,4177**, F1 **0,4487**. XGBoost optimizado `xgb_unweighted` mejora AP a **0,4832** en `full`, pero reduce recall a **0,0833** con umbral 0,5. Todos son **resultados de validación exploratoria**.

**Lo no logrado todavía:** meta académica **ROC-AUC ≥ 0,80 en prueba**, evaluación final del holdout, aprobación científica/operativa del modelo, calibración de probabilidad/bandas, APIs FastAPI/Fastify, frontend Next.js, validación de usuarios y despliegue.

**Decisión de continuidad:** **cerrar por ahora la búsqueda de hiperparámetros y avanzar con el prototipo de FastAPI**, preservando el estado exploratorio y la posibilidad de sustituir/reentrenar el pipeline posteriormente. Evitar confundir software funcionando con tesis validada. El registro de pendientes permitirá retomar la revisión científica y el sistema completo durante 2027.

---

**Fin de la memoria técnica — versión 1.0.**
