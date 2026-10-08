# DEPI No-Show ML

Proyecto Python de la tesis **Sistema web predictivo basado en XGBoost y Random Forest para el pronóstico de inasistencias en una empresa del rubro de estética y cuidado personal**.

La fase 1 valida y exporta la tabla analítica existente de FLOWww. La extracción usa exclusivamente PostgreSQL y archivos locales. No ejecuta ETL, modifica datos, entrena modelos ni implementa una API. Los módulos futuros se incorporarán en este mismo paquete cuando se autorice la siguiente fase.

```text
FLOWww → raw_flow → analytics.appointment_training_dataset_v1
                                  ↓ transacción de solo lectura
                           depi-ml inspect-dataset
                           depi-ml export-dataset
                                  ↓
                      CSV + manifest + reporte de calidad
```

## Estructura

```text
src/depi_ml/
  cli.py                  Comandos de inspección y exportación
  config.py               Configuración desde entorno y .env
  db/connection.py        Instantánea REPEATABLE READ de solo lectura
  datasets/schema.py      Contrato único de columnas y roles
  datasets/validator.py   Validaciones y estadísticas SQL
  datasets/exporter.py    Streaming COPY, verificación y publicación
data/exports/             Archivos locales excluidos de Git
docs/dataset_contract.md  Contrato y limitaciones metodológicas
tests/                    Pruebas unitarias sin PostgreSQL
prompts/PROMPT.md         Especificación original
bds/                     SQL existente del usuario; el exportador no lo ejecuta
docker-compose.yml        PostgreSQL externo existente
```

## Instalación

Requiere **Python 3.12 o superior**, PostgreSQL 18.4 accesible y permisos `USAGE` del schema y `SELECT` sobre todas las columnas del contrato. Dependencias: psycopg 3 con distribución binaria y python-dotenv; pytest para desarrollo. No necesita pandas ni dependencias de entrenamiento.

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
Copy-Item .env.example .env
```

Sin activar el entorno, utiliza `.venv/bin/depi-ml` en Linux/macOS o `.\.venv\Scripts\depi-ml.exe` en Windows. Si PowerShell impide activar scripts, puedes invocar directamente `.\.venv\Scripts\python.exe` y el ejecutable CLI.

## Configuración de la base externa

Edita `.env` con tus credenciales reales. El ejemplo usa `depi_noshow` y `depi_admin` según el prompt. El `docker-compose.yml` existente en este repositorio configura **`depi-noshow-db` y `admin`**; usa esos valores si trabajas con ese contenedor. No se cambia la configuración existente.

```dotenv
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=depi-noshow-db
POSTGRES_USER=admin
POSTGRES_PASSWORD=<tu contraseña local>
DATASET_SCHEMA=analytics
DATASET_TABLE=appointment_training_dataset_v1
DATASET_EXPORT_DIR=data/exports
DATASET_BATCH_SIZE=10000
```

El CLI lee `.env` desde el directorio de ejecución; las variables del entorno tienen prioridad. Las rutas relativas también se resuelven desde ese directorio. La contraseña debe configurarse y `CHANGE_ME` se rechaza.

Python en la máquina anfitriona se conecta al **puerto publicado**, que puedes consultar sin ver credenciales:

```bash
docker compose ps
docker compose port postgres 5432
```

Por ejemplo, para `0.0.0.0:5544`, configura `POSTGRES_PORT=5544`. No asumas que es 5432. Si Python está en otro contenedor de la misma red Compose, utiliza el host `postgres` y el puerto interno 5432. PostgreSQL se administra por separado; no hace falta reiniciarlo para exportar ni ejecutar los SQL de `bds/` nuevamente.

## Uso

Desde la raíz del repositorio, con el entorno virtual activo:

```bash
depi-ml inspect-dataset
depi-ml export-dataset
depi-ml export-dataset --output data/exports/training_dataset_v1.csv --batch-size 10000
```

También está disponible `python -m depi_ml.cli`. `inspect-dataset` muestra el reporte JSON con estructura, verificaciones y estadísticas; no crea archivos. Los errores críticos terminan con código 1; argumentos inválidos, con código 2; una cancelación, con 130. Una exportación correcta termina con código 0 y muestra filas, bytes, clases y rutas.

La exportación usa [COPY TO STDOUT por bloques de psycopg](https://www.psycopg.org/psycopg3/docs/basic/copy.html#copying-block-by-block). PostgreSQL decide el tamaño de los bloques de red. **`--batch-size` indica bytes del buffer del archivo local**, no filas por consulta ni el tamaño del paquete PostgreSQL; además determina el intervalo de filas de los mensajes de verificación. No hay cursor `fetchmany()`, muestreo ni `LIMIT`. La memoria del cliente queda acotada por un bloque COPY, un registro CSV y el buffer de escritura, además de los pequeños agregados del reporte. PostgreSQL puede necesitar memoria o espacio temporal para ordenar y calcular distintos.

## CSV y artefactos

La salida predeterminada es:

```text
data/exports/training_dataset_v1.csv
data/exports/training_dataset_v1_manifest.json
data/exports/training_dataset_v1_quality_report.json
```

Un destino personalizado cambia también los nombres de los reportes usando el mismo nombre base. El CSV tiene 27 columnas en el orden definido en `datasets/schema.py`, conserva las 19 predictoras y los 8 campos de trazabilidad, objetivo, fechas, auditoría y metadatos. Incluye **todas** las filas y nulos; no imputa ni recodifica. Ordena por `appointment_at, appointment_id`.

Utiliza UTF-8, coma y encabezados. Los nulos son campos vacíos sin comillas; las cadenas vacías PostgreSQL se conservan como `""`. Los booleanos mantienen la representación PostgreSQL `t`/`f`. Los timestamps se representan en UTC con `T` y offset `+00`, conservando precisión disponible; únicamente cambia su representación textual. Véase el [contrato completo](docs/dataset_contract.md).

Conteo, estadísticas y COPY comparten una transacción `REPEATABLE READ READ ONLY`. Después se analiza el CSV registro por registro, incluidos saltos de línea dentro de campos entrecomillados, se compara su total y clases con PostgreSQL y se calcula SHA-256. El manifest registra columnas, origen, versiones de features, orden, estrategia, tamaño y hash; el reporte de calidad contiene todas las estadísticas y comprobaciones.

Los tres artefactos se preparan en un directorio temporal del mismo filesystem; se publican los reportes y finalmente se reemplaza atómicamente el CSV. Si ocurre un error recuperable durante la publicación, se restauran los destinos anteriores. La transacción debe cerrar correctamente antes de publicar. Una caída del sistema durante la sustitución de varios archivos puede dejar reportes de otra ejecución: los tres archivos no constituyen una única operación atómica del filesystem. Comprueba el hash del CSV contra el manifest y vuelve a exportar en ese caso. Evita exportaciones concurrentes al mismo destino.

## Validación y calidad

Se detiene ante falta de acceso, schema/tabla inexistente, permisos insuficientes, columnas ausentes, tipos incompatibles, tabla vacía, `target` inválido o nulo, IDs de cita duplicados o nulos, IDs de cliente nulos, fechas de predicción/cita ausentes, timestamps infinitos o `prediction_at >= appointment_at`. El control de tipos acepta familias de enteros, flotantes y texto compatibles; exige booleanos y timestamps con zona horaria. No admite `timestamp` sin zona ni conversiones implícitas de texto a números.

El reporte describe total, asistencias, no-shows y porcentaje; fechas extremas; clientes y clínicas distintos; nulos de cada columna; casos del tracking `-1`; distribución de `feature_version`; anomalías de edad y anticipación. Edad negativa o mayor a 120 y anticipación negativa o no finita generan alertas descriptivas: **no excluyen ni modifican registros**. Los nulos de predictoras se conservan. Ningún log muestra filas de clientes o contraseñas.

Para verificar integridad del CSV:

```bash
python -c "from pathlib import Path; import hashlib,json; p=Path('data/exports/training_dataset_v1.csv'); h=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(); m=json.loads(p.with_name(p.stem+'_manifest.json').read_text()); print(h == m['sha256'])"
```

## Pruebas

```bash
python -m pytest
depi-ml --help
python -c "import depi_ml; import depi_ml.datasets.exporter"
```

Las pruebas no conectan a PostgreSQL. Utilizan fixtures mínimas de prueba, nunca datos de clientes ni un dataset de producción simulado. Cubren configuración, contrato y tipos, clases, duplicados, temporalidad, nulos, reportes, streaming, integridad CSV, redacción de errores y preservación de archivos ante fallos. Para comprobar la integración real, ejecuta `inspect-dataset` y `export-dataset` con tu `.env` local.

Si el anfitrión solo dispone de Python 3.10, instala Python 3.12+ para crear `.venv`. Alternativamente puedes verificar la instalación y las pruebas con Docker sin modificar la base:

```bash
docker run --rm -v "$PWD:/app" -w /app python:3.12-slim sh -c 'python -m venv .venv-docker && .venv-docker/bin/python -m pip install -e ".[dev]" && .venv-docker/bin/python -m pytest && .venv-docker/bin/depi-ml --help'
```

El entorno `.venv-docker` es de Linux y debe utilizarse dentro del contenedor. No puede activarse en macOS/Windows. El contenedor de verificación requiere acceso de red para descargar las dependencias.

## Problemas frecuentes

- **Conexión rechazada o timeout:** revisa `docker compose ps`, puerto publicado, host y firewall. `127.0.0.1` dentro de un contenedor identifica ese contenedor.
- **Autenticación o base inexistente:** confirma los nombres reales del Compose y tu contraseña; no son necesariamente los de `.env.example`.
- **Permisos:** solicita al administrador `USAGE` del schema y `SELECT` de las columnas. El exportador no otorga permisos ni ejecuta DDL.
- **Faltan tabla o columnas:** confirma que la preparación SQL ya se realizó en la base correcta. El CLI no reconstruye ETL.
- **Validación temporal o target inválido:** revisa el origen analítico. No se generan CSV definitivos de una ejecución fallida ni se corrigen los datos automáticamente.
- **No existe `depi-ml`:** activa el entorno e instala con `python -m pip install -e '.[dev]'`.
- **Error de escritura:** revisa espacio libre y permisos del destino; debe admitir temporales y renombrados.

## Privacidad y próximas fases

`.env`, los datasets, reportes y modelos están excluidos de Git. No subas archivos de clientes ni credenciales; el Compose existente debe revisarse antes de publicarlo, porque contiene configuración sensible previa a esta implementación. El exportador no envía datos a APIs externas.

En futuras fases autorizadas se incorporarán Random Forest/XGBoost, división temporal, evaluación y calibración, SHAP y versionado; después, FastAPI para inferencia individual y por lotes. Se reutilizarán el contrato de features y el preprocesamiento aprendido en este mismo repositorio. Las validaciones técnicas de esta fase no acreditan aún la validez científica del dataset.
