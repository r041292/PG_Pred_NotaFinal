import pandas as pd

ruta_prereq = "para trabajar pre req progv2.xlsx"
ruta_historial = "detalle matricula sistemas pedacito.xlsx"

prereqs= pd.read_excel(ruta_prereq)
historial = pd.read_excel(ruta_historial)

prereqs = prereqs.rename(columns={"Smbarul_Key_Rule": "CodigoAsignatura", "prereq completo": "Reglas"})
historial= historial.rename(columns={"Cod materia curso": "CodigoAsignatura", "Pidm": "EstudianteID", "Calificacion_Final": "Nota", "Periodo": "Periodo"})

# ==============================
# 1. Cargar los datos
# ==============================
# historial: registros de materias cursadas
# prereqs: reglas de prerrequisitos
# (aquí asumo que ya tienes los DataFrames cargados)

# Ejemplo de columnas esperadas:
# historial: EstudianteID | CodigoAsignatura | Nota | Periodo
# prereqs:   CodigoAsignatura | Reglas (ej: "MAT101 AND (FIS100 OR FIS200)")

# ==============================
# 2. Preprocesar reglas de prerrequisitos
# ==============================

def expand_prereqs(df, col="Reglas"):
    """
    Convierte las reglas de prerrequisitos en un formato largo (long format).
    Maneja separadores como AND / OR y listas separadas por comas.
    """
    # Paso 1: Normalizar separadores
    df = df.copy()
    df[col] = df[col].str.replace(r"\(", "", regex=True).str.replace(r"\)", "", regex=True)
    
    # Paso 2: Separar por AND primero
    df = df.assign(AND_parts=df[col].str.split("AND"))
    df = df.explode("AND_parts")
    df["AND_parts"] = df["AND_parts"].str.strip()

    # Paso 3: Dentro de cada AND, separar opciones OR
    df = df.assign(OR_parts=df["AND_parts"].str.split("OR"))
    df = df.explode("OR_parts")
    df["OR_parts"] = df["OR_parts"].str.strip()

    # Resultado: cada fila es un prerrequisito simple
    return df[["CodigoAsignatura", "AND_parts", "OR_parts"]].rename(
        columns={"AND_parts": "GrupoAND", "OR_parts": "Prerrequisito"}
    )

prereqs_long = expand_prereqs(prereqs)

# ==============================
# 3. Unir historial con prerrequisitos
# ==============================
# Hacemos merge entre historial y prereqs_long para ver qué cursos del historial
# cumplen con los prerrequisitos de cada asignatura.

hist_prereq = prereqs_long.merge(
    historial,
    left_on="Prerrequisito",
    right_on="CodigoAsignatura",
    how="left",
    suffixes=("_req", "_hist")
)

# ==============================
# 4. Validar cumplimiento
# ==============================
# Regla: solo cuentan los cursos del historial con Periodo <= al curso objetivo.
# (Aquí asumimos que luego cruzas con las inscripciones actuales del estudiante.)

# Supongamos que tienes otro DF "inscripciones":
# inscripciones: EstudianteID | CodigoAsignatura | Periodo

inscripciones = pd.read_excel(ruta_historial)
inscripciones= inscripciones.rename(columns={"Cod materia curso": "CodigoAsignatura", "Pidm": "EstudianteID", "Calificacion_Final": "Nota", "Periodo": "Periodo"})

validaciones = inscripciones.merge(
    hist_prereq,
    left_on="CodigoAsignatura",
    right_on="CodigoAsignatura_req",
    how="left"
)

# Mantener solo intentos previos
validaciones = validaciones[
    validaciones["Periodo_hist"] <= validaciones["Periodo"]
]

# ==============================
# 5. Reglas AND / OR
# ==============================
# Agrupamos para verificar:
# - Que al menos 1 curso en cada grupo OR esté aprobado
# - Que todos los grupos AND estén satisfechos

# Primero, marcar si el prerrequisito fue aprobado
validaciones["Aprobado"] = validaciones["Nota"] >= 3  # Ajusta el criterio según tu regla

# Evaluar por grupo
or_check = (
    validaciones.groupby(
        ["EstudianteID", "CodigoAsignatura", "GrupoAND"]
    )["Aprobado"]
    .any()
    .reset_index()
)

# Luego, AND = todos los grupos deben ser True
and_check = (
    or_check.groupby(["EstudianteID", "CodigoAsignatura"])["Aprobado"]
    .all()
    .reset_index()
    .rename(columns={"Aprobado": "CumplePrerrequisito"})
)

# ==============================
# 6. Resultado final
# ==============================
resultado = inscripciones.merge(and_check, on=["EstudianteID", "CodigoAsignatura"], how="left")
resultado["CumplePrerrequisito"] = resultado["CumplePrerrequisito"].fillna(False)

# ==============================
# OUTPUT
# ==============================
print(resultado)
