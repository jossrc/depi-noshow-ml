# Guía local: levantar el proyecto y regenerar el CSV

Esta guía es para **probar la Fase 1** del proyecto: conectarte a PostgreSQL, validar la tabla analítica y volver a generar el dataset CSV.

No se entrenan modelos. No se levanta FastAPI. No se reconstruyen tablas SQL.

Estás en macOS. En esta máquina ya hay **Python 3.12.15** y el contenedor `depi_noshow` en el puerto **5432**.

---

## 1. Qué hace este proyecto (mapa mental)

```text
PostgreSQL (Docker)
  schema analytics
    tabla appointment_training_dataset_v1
              │
              │  depi-ml inspect-dataset   ← solo lee y muestra estadísticas
              │  depi-ml export-dataset    ← lee y escribe archivos locales
              ▼
data/exports/
  training_dataset_v1.csv
  training_dataset_v1_manifest.json
  training_dataset_v1_quality_report.json
```

El CSV **no es la fuente de verdad**. La fuente es PostgreSQL. Si borras el CSV, el exportador lo vuelve a crear desde la tabla.

Piensa en Python como en otros lenguajes:

| Concepto que ya conoces | Equivalente aquí |
|---|---|
| `node_modules` / `vendor` | `.venv/` (entorno virtual) |
| `npm install` / `composer install` | `pip install -e '.[dev]'` |
| `.env` de Laravel o Node | `.env` (credenciales de Postgres) |
| CLI del proyecto (`artisan`, `bin/console`) | `depi-ml` |

El **entorno virtual** (`.venv`) es una carpeta local con su propio Python y sus librerías. Así no mezclas dependencias con el resto de tu Mac.

---

## 2. Requisitos

Antes de empezar, abre una terminal y confirma:

```bash
cd /Users/joss/Personal/tesis/depi-noshow-ml

python3.12 --version
# Esperado: Python 3.12.x

docker ps --filter name=depi_noshow
# Esperado: STATUS "Up ... (healthy)" y puerto 5432
```

Si el contenedor no está arriba:

```bash
docker start depi_noshow
```

Si `python3.12` no existe, instala Python 3.12 o superior. Este proyecto **no funciona** con 3.10 o 3.11.

---

## 3. Crear el entorno virtual (solo la primera vez)

Desde la raíz del repositorio:

```bash
cd /Users/joss/Personal/tesis/depi-noshow-ml

python3.12 -m venv .venv
```

Eso crea la carpeta `.venv/`. No la subas a Git; ya está ignorada.

Actívalo (hay que hacerlo **en cada terminal nueva**):

```bash
source .venv/bin/activate
```

Cuando está activo, el prompt suele mostrar `(.venv)` al inicio. Comprueba:

```bash
which python
python --version
# which debe apuntar a .../depi-noshow-ml/.venv/bin/python
# versión: Python 3.12.x
```

Para salir del entorno más tarde:

```bash
deactivate
```

---

## 4. Instalar el proyecto (solo la primera vez)

Con el entorno **activo**:

```bash
python -m pip install -e '.[dev]'
```

Qué significa:

- `pip` instala dependencias (`psycopg`, `python-dotenv`, `pytest`).
- `-e` (editable) instala el paquete `depi-ml` apuntando a tu código. Si editas `src/`, no hace falta reinstalar.
- `.[dev]` incluye también las herramientas de prueba.

Comprueba que el comando existe:

```bash
depi-ml --help
```

Debes ver los subcomandos `inspect-dataset` y `export-dataset`.

Si aparece `command not found: depi-ml`:

1. Confirma que el entorno está activo (`which python` apunta a `.venv`).
2. Vuelve a ejecutar `python -m pip install -e '.[dev]'`.
3. Alternativa sin el comando corto:

```bash
python -m depi_ml.cli --help
```

---

## 5. Configurar la conexión (`.env`)

El proyecto **no** lleva la contraseña en el código. Lee un archivo `.env` en la raíz.

Si **ya tienes** `.env` (en este repo suele existir porque ya se exportó una vez), no lo regeneres. Ábrelo y verifica que coincida con tu Docker.

Si **no** existe:

```bash
cp .env.example .env
```

Luego edita `.env`. Con el `docker-compose.yml` de este repo, los valores reales suelen ser:

```dotenv
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
POSTGRES_DB=depi-noshow-db
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin

DATASET_SCHEMA=analytics
DATASET_TABLE=appointment_training_dataset_v1
DATASET_EXPORT_DIR=data/exports
DATASET_BATCH_SIZE=10000
```

Notas importantes:

- El `.env.example` usa otros nombres (`depi_noshow` / `depi_admin`). **Usa los de tu Docker**, no los del ejemplo.
- No dejes `POSTGRES_PASSWORD=CHANGE_ME`; el CLI lo rechaza.
- No subas `.env` a Git.
- Ejecuta siempre los comandos desde la **raíz del repositorio**, porque el CLI busca `.env` ahí.

---

## 6. Probar que Python ve PostgreSQL (sin generar archivos)

Con el entorno activo y desde la raíz:

```bash
depi-ml inspect-dataset
```

Esto solo **lee**. No crea ni borra CSV.

Si funciona, verás un JSON grande con:

- `"status": "passed"`
- `"row_count"` cercano a `213873`
- validaciones en `true`
- conteos de asistencias y no-shows

Si falla, no pases al export. Revisa la [sección 9](#9-errores-frecuentes).

---

## 7. Borrar el CSV y volver a generarlo

Este es el experimento que quieres hacer. El exportador lee otra vez la tabla y escribe archivos nuevos.

### 7.1. Borrar los tres artefactos

```bash
rm data/exports/training_dataset_v1.csv \
   data/exports/training_dataset_v1_manifest.json \
   data/exports/training_dataset_v1_quality_report.json
```

Comprueba que desaparecieron:

```bash
ls data/exports/
# Solo debería quedar .gitkeep (o la carpeta vacía de CSV/JSON)
```

Borrar el CSV **no borra** los datos de PostgreSQL.

### 7.2. Exportar

```bash
depi-ml export-dataset
```

Tarda unos segundos o un par de minutos (son ~214 mil filas y ~39 MB). Verás logs de progreso y, al final, algo como:

```text
Exportación completada: 213873 filas, 40869682 bytes.
Asistencias: 169553; candidatos no-show: 44320.
CSV: data/exports/training_dataset_v1.csv
Manifest: data/exports/training_dataset_v1_manifest.json
Calidad: data/exports/training_dataset_v1_quality_report.json
```

Los números pueden variar un poco si la tabla analítica se reconstruyó después. Lo importante es que termine con código 0 y cree los tres archivos.

Opcional, con ruta y tamaño de buffer explícitos:

```bash
depi-ml export-dataset \
  --output data/exports/training_dataset_v1.csv \
  --batch-size 10000
```

`--batch-size` no limita filas. Es el tamaño del buffer local al escribir el archivo.

---

## 8. Cómo saber que salió bien

```bash
ls -lh data/exports/
```

Deben existir:

```text
training_dataset_v1.csv
training_dataset_v1_manifest.json
training_dataset_v1_quality_report.json
```

Comprobaciones rápidas:

```bash
# Filas del CSV = encabezado + datos
wc -l data/exports/training_dataset_v1.csv
# Esperado: 213874  (1 línea de header + 213873 registros)

# Primeras líneas
head -n 3 data/exports/training_dataset_v1.csv
```

El hash del CSV debe coincidir con el manifest:

```bash
python -c "from pathlib import Path; import hashlib, json; p=Path('data/exports/training_dataset_v1.csv'); h=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest(); m=json.loads(Path('data/exports/training_dataset_v1_manifest.json').read_text()); print('igual' if h == m['sha256'] else 'DISTINTO'); print(h); print(m['row_count'])"
```

También puedes abrir el reporte de calidad y buscar:

- `"status": "passed"`
- `"csv_row_count_matches": true`

Si la exportación falla a mitad, **no** debe quedar un CSV incompleto presentado como válido. El proceso escribe primero archivos temporales y solo al final los publica.

---

## 9. Errores frecuentes

| Qué ves | Qué revisar |
|---|---|
| `command not found: depi-ml` | Activa `.venv` e instala con `pip install -e '.[dev]'`. |
| `Falta configurar POSTGRES_PASSWORD` | No existe `.env` o está vacío. Cópialo y edítalo. |
| `Reemplaza POSTGRES_PASSWORD` | Todavía dice `CHANGE_ME`. |
| Autenticación rechazada | Usuario/contraseña/base no coinciden con Docker. Usa `depi-noshow-db` / `admin`. |
| Conexión rechazada o timeout | `docker ps`: el contenedor debe estar `healthy` en `5432`. |
| Schema o tabla no existen | La preparación SQL ya debía estar hecha. Este CLI **no** ejecuta los scripts de `bds/`. |
| `No existe depi-ml` después de cerrar la terminal | Cada terminal nueva necesita `source .venv/bin/activate`. |

El exportador es **solo lectura**. No hace `INSERT`, `UPDATE`, `DELETE` ni reconstruye ETL.

---

## 10. Pruebas unitarias (opcional)

No necesitan PostgreSQL ni el CSV. Sirven para comprobar que el código del exportador está sano:

```bash
source .venv/bin/activate
python -m pytest
```

Si pasan, el paquete está bien instalado. Eso **no** sustituye `inspect-dataset` / `export-dataset` contra tu base real.

---

## 11. Rutina corta (cuando ya lo instalaste una vez)

```bash
cd /Users/joss/Personal/tesis/depi-noshow-ml
source .venv/bin/activate
docker start depi_noshow   # solo si no está corriendo

depi-ml inspect-dataset
depi-ml export-dataset
```

Para repetir el experimento de borrar y regenerar, inserta el `rm` de la [sección 7.1](#71-borrar-los-tres-artefactos) entre `inspect-dataset` y `export-dataset`.

---

## 12. Qué no hacer

- No subas el CSV, el manifest, el reporte ni el `.env` a GitHub.
- No ejecutes de nuevo los SQL de `bds/` para “regenerar el CSV”. Eso es otra etapa (construir la tabla analítica).
- No hace falta pandas, Jupyter ni entrenar modelos para esta prueba.
- No uses el Python global de macOS (`/usr/bin/python3`) si no es 3.12. Siempre el de `.venv`.
