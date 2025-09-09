import numpy as np
import pywt
from scipy.signal import hilbert
from scipy.ndimage import binary_closing, binary_opening
from typing import List, Tuple

# =============================================================================
# SECCIÓN 1: UTILIDADES BÁSICAS
# =============================================================================

def muestras_por_ciclo(fs: int, f_nom: float = 60.0) -> int:
    """Número de muestras por ciclo de la señal de red."""
    return int(round(fs / f_nom))

def convertir_mascara_a_intervalos(mascara: np.ndarray) -> List[Tuple[int, int]]:
    """Convierte una máscara booleana en intervalos (ini, fin)."""
    if mascara.size == 0 or not mascara.any():
        return []
    m = mascara.astype(np.int8)
    dm = np.diff(np.r_[0, m, 0])
    inicios = np.flatnonzero(dm == 1)
    finales = np.flatnonzero(dm == -1)
    return list(zip(inicios, finales))

def unir_intervalos(intervalos: List[Tuple[int, int]], margen_muestras: int) -> List[Tuple[int, int]]:
    """Une intervalos cercanos o solapados (separados por menos de margen_muestras)."""
    if not intervalos:
        return []
    intervalos.sort(key=lambda x: x[0])
    union = [intervalos[0]]
    for ini, fin in intervalos[1:]:
        u_ini, u_fin = union[-1]
        if ini <= (u_fin + margen_muestras):
            union[-1] = (u_ini, max(u_fin, fin))
        else:
            union.append((ini, fin))
    return union

def postprocesar_mascara(mascara: np.ndarray, mpc: int) -> np.ndarray:
    """Suaviza la máscara eliminando picos y rellenando huecos pequeños."""
    se_cierre = np.ones(max(1, mpc // 4), dtype=bool)  # cierra huecos cortos
    se_apertura = np.ones(max(1, mpc // 8), dtype=bool)  # elimina picos breves
    m = binary_closing(mascara, structure=se_cierre)
    m = binary_opening(m, structure=se_apertura)
    return m
# =============================================================================
# SECCIÓN 2: GENERACIÓN DE MÁSCARAS (TRIGGERS)
# =============================================================================

def generar_mascara_rms(senal: np.ndarray, fs: int, f_nom: float = 60.0,
                        umbral_sag: float = 0.90, percentil_ref: float = 95.0) -> np.ndarray:
    """Máscara basada en RMS por medio ciclo con solapamiento 50%."""
    mpc = muestras_por_ciclo(fs, f_nom)
    hop = mpc // 2
    if hop <= 0:
        return np.zeros(len(senal), dtype=bool)

    rms_vals, centros = [], []
    for ini in range(0, len(senal) - hop + 1, hop):
        fin = min(ini + mpc, len(senal))
        if fin - ini < hop: break
        rms_vals.append(np.sqrt(np.mean(senal[ini:fin]**2)))
        centros.append((ini + fin) // 2)

    if not rms_vals:
        return np.zeros(len(senal), dtype=bool)

    v_nominal = np.percentile(rms_vals, percentil_ref)
    umbral = umbral_sag * v_nominal

    mascara = np.zeros(len(senal), dtype=bool)
    for centro, rms_val in zip(centros, rms_vals):
        if rms_val < umbral:
            ini_p = max(0, centro - hop)
            fin_p = min(len(senal), centro + hop)
            mascara[ini_p:fin_p] = True
    
    return postprocesar_mascara(mascara, mpc)
    

def generar_mascara_swt(senal: np.ndarray, wavelet: str = 'db4', niveles: int = 4) -> np.ndarray:
    """Máscara basada en energía de la SWT (Wavelet Estacionaria)."""
    coefs = pywt.swt(senal, wavelet, level=niveles)
    energia = np.sum([cD**2 for (_, cD) in coefs], axis=0)

    umbral = np.percentile(energia, 99.0)

    return postprocesar_mascara(energia > umbral, muestras_por_ciclo(7200))

def generar_mascara_hilbert(senal: np.ndarray, percentil_ref: float = 95.0,
                            umbral_sag: float = 0.90, umbral_swell: float = 1.10) -> np.ndarray:
    """Máscara basada en la envolvente de Hilbert."""
    envolvente = np.abs(hilbert(senal))
    v_ref = np.percentile(envolvente, percentil_ref)
    if v_ref < 1e-9:
        return np.zeros_like(envolvente, dtype=bool)

    caidas = envolvente < (umbral_sag * v_ref)
    subidas = envolvente > (umbral_swell * v_ref)
    return postprocesar_mascara(caidas | subidas, muestras_por_ciclo(7200))

# =============================================================================
# SECCIÓN 3: DETECTORES DE EVENTOS INDIVIDUALES
# =============================================================================

def detectar_eventos_rms(senal: np.ndarray, fs: int, umbral_sag: float = 0.9,
                         ciclos_minimos: float = 0.5) -> List[Tuple[int, int]]:
    """Detecta eventos usando solo RMS."""
    mpc = muestras_por_ciclo(fs)
    mascara = generar_mascara_rms(senal, fs, umbral_sag=umbral_sag)
    intervalos = convertir_mascara_a_intervalos(mascara)
    min_duracion = int(ciclos_minimos * mpc)
    return [(i, f) for i, f in intervalos if (f - i) >= min_duracion]

def detectar_eventos_hilbert(senal: np.ndarray, fs: int, umbral_sag: float = 0.9,
                             umbral_swell: float = 1.1, ciclos_minimos: float = 0.5) -> List[Tuple[int, int]]:
    """Detecta eventos usando solo Hilbert."""
    mpc = muestras_por_ciclo(fs)
    mascara = generar_mascara_hilbert(senal, umbral_sag=umbral_sag, umbral_swell=umbral_swell)
    intervalos = convertir_mascara_a_intervalos(mascara)
    min_duracion = int(ciclos_minimos * mpc)
    return [(i, f) for i, f in intervalos if (f - i) >= min_duracion]

def detectar_eventos_swt(senal: np.ndarray, fs: int, ciclos_minimos: float = 0.1) -> List[Tuple[int, int]]:
    """Detecta eventos usando solo SWT."""
    mpc = muestras_por_ciclo(fs)
    mascara = generar_mascara_swt(senal)
    intervalos = convertir_mascara_a_intervalos(mascara)
    min_duracion = int(ciclos_minimos * mpc)
    return [(i, f) for i, f in intervalos if (f - i) >= min_duracion]

# =============================================================================
# SECCIÓN 4: DETECTOR AVANZADO CON FUSIÓN
# =============================================================================
def refinar_bordes_con_swt(
    eventos_prelim: List[Tuple[int, int]],
    intervalos_swt: List[Tuple[int, int]],
    fs: int,
    n_total: int,
    f_nom: float = 60.0,
    ventana_busqueda_ciclos: float = 0.75
) -> List[Tuple[int, int]]:
    """
    Refina los bordes de eventos preliminares usando la lógica correcta:
    - Inicio: Se alinea con el ÚLTIMO transitorio SWT antes del borde.
    - Fin: Se alinea con el PRIMER transitorio SWT después del borde.
    """
    if not intervalos_swt or not eventos_prelim:
        return eventos_prelim # No hay nada que refinar

    mpc = muestras_por_ciclo(fs, f_nom)
    ventana_muestras = int(ventana_busqueda_ciclos * mpc)

    # Crear un solo arreglo con todos los puntos de cambio de SWT (inicios y fines)
    puntos_swt = sorted(list(set([p for a, b in intervalos_swt for p in (a, b)])))
    puntos_swt = np.array(puntos_swt)

    eventos_refinados = []
    for ini_aprox, fin_aprox in eventos_prelim:
        
        # --- Refinar INICIO ---
        # Ventana de búsqueda: desde mucho antes hasta un poco después del inicio aprox.
        lim_inf_ini = ini_aprox - ventana_muestras
        lim_sup_ini = ini_aprox + int(0.1 * mpc) # Pequeña tolerancia hacia adelante
        
        # Candidatos: puntos SWT que caen ANTES del límite superior de la ventana
        candidatos_ini = puntos_swt[puntos_swt <= lim_sup_ini]
        
        # De esos, nos quedamos con el que esté MÁS CERCA del inicio (el último)
        if candidatos_ini.size > 0:
            nuevo_inicio = max(candidatos_ini) # El ÚLTIMO transitorio antes
        else:
            nuevo_inicio = ini_aprox # Fallback: no se encontró candidato

        # --- Refinar FIN ---
        # Ventana de búsqueda: desde un poco antes del fin hasta mucho después
        lim_inf_fin = fin_aprox - int(0.1 * mpc) # Pequeña tolerancia hacia atrás
        lim_sup_fin = fin_aprox + ventana_muestras
        
        # Candidatos: puntos SWT que caen DESPUÉS del límite inferior de la ventana
        candidatos_fin = puntos_swt[puntos_swt >= lim_inf_fin]
        
        # De esos, nos quedamos con el que esté MÁS CERCA del fin (el primero)
        if candidatos_fin.size > 0:
            nuevo_fin = min(candidatos_fin) # El PRIMER transitorio después
        else:
            nuevo_fin = fin_aprox # Fallback: no se encontró candidato

        eventos_refinados.append((
            max(0, nuevo_inicio),
            min(n_total, nuevo_fin)
        ))

    return eventos_refinados
def detectar_eventos_con_fusion(senal: np.ndarray, fs: int, umbral_sag: float = 0.9,
                                umbral_swell: float = 1.1, ciclos_minimos: float = 0.5) -> List[Tuple[int, int]]:
    """Fusiona RMS, SWT y Hilbert para una detección más robusta."""
    mpc = muestras_por_ciclo(fs)
    medio_ciclo = mpc // 2
    if medio_ciclo <= 0:
        return []

    # Máscaras individuales
    rms_mask = generar_mascara_rms(senal, fs, umbral_sag=umbral_sag)
    swt_mask = generar_mascara_swt(senal)
    hilbert_mask = generar_mascara_hilbert(senal, umbral_sag=umbral_sag, umbral_swell=umbral_swell)

    # Coincidencias SWT & Hilbert
    apoyo = swt_mask & hilbert_mask
    intervalos_apoyo = convertir_mascara_a_intervalos(apoyo)
    apoyo_filtrado = np.zeros_like(apoyo)
    for ini, fin in intervalos_apoyo:
        if (fin - ini) >= max(1, mpc // 4):
            apoyo_filtrado[ini:fin] = True

    # Lógica: RMS obligatorio + apoyo SWT/Hilbert
    activacion = rms_mask & (swt_mask | hilbert_mask | apoyo_filtrado)
    preliminares = convertir_mascara_a_intervalos(activacion)
    intervalos_swt = convertir_mascara_a_intervalos(swt_mask)
    # Refinar bordes
    refinados = refinar_bordes_con_swt(
        preliminares,
        intervalos_swt,
        fs=fs,
        n_total=len(senal),
        ventana_busqueda_ciclos=0.75,
    )

    # Unir y filtrar
    unidos = unir_intervalos(refinados, margen_muestras=medio_ciclo)
    min_duracion = int(ciclos_minimos * mpc)
    return [(i, f) for i, f in unidos if (f - i) >= min_duracion]
