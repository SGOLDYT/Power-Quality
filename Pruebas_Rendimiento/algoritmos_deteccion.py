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
    hop = mpc  // 2  # salto de medio ciclo
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
def detectar_eventos_con_fusion(senal: np.ndarray, fs: int, umbral_sag: float = 0.9,
                                umbral_swell: float = 1.1, ciclos_minimos: float = 0.5) -> List[Tuple[int, int]]:
    """
    Usa RMS como Región de Interés (ROI) para filtrar a 
    Hilbert y SWT, y luego refina los bordes del RMS con la mediana de las 
    sugerencias de los asistentes dentro de la ROI.
    """
    mpc = muestras_por_ciclo(fs)
    
    # Se generan las tres máscaras base.
    rms_mask = generar_mascara_rms(senal, fs, umbral_sag=umbral_sag)
    swt_mask = generar_mascara_swt(senal)
    hilbert_mask = generar_mascara_hilbert(senal, umbral_sag=umbral_sag, umbral_swell=umbral_swell)

    # Se filtran las máscaras de los asistentes para que solo existan DENTRO de la máscara del RMS.
    h_mask_filtrada = hilbert_mask & rms_mask
    w_mask_filtrada = swt_mask & rms_mask
    
    # Se extraen todos los puntos de borde (inicio y fin) de los asistentes ya filtrados.
    puntos_h = convertir_mascara_a_intervalos(h_mask_filtrada)
    puntos_w = convertir_mascara_a_intervalos(w_mask_filtrada)
    
    # Creamos dos listas: una con todas las sugerencias de INICIO y otra con las de FIN.
    inicios_sugeridos = np.array(sorted([ini for ini, fin in puntos_h] + [ini for ini, fin in puntos_w]))
    fines_sugeridos = np.array(sorted([fin for ini, fin in puntos_h] + [fin for ini, fin in puntos_w]))


    # Se obtienen los intervalos del RMS, que son nuestra guía principal.
    intervalos_guia_rms = convertir_mascara_a_intervalos(rms_mask)
    eventos_refinados = []
    
    for ini_rms, fin_rms in intervalos_guia_rms:
        # Definimos una pequeña ventana de búsqueda alrededor de los bordes del RMS.
        ventana = mpc 
        
        # Para el INICIO:
        # Buscamos todas las sugerencias de inicio que estén cerca del inicio del RMS.
        candidatos_ini = inicios_sugeridos[(inicios_sugeridos >= ini_rms - ventana) & 
                                            (inicios_sugeridos <= ini_rms + ventana)]
        # El nuevo inicio será la mediana de esas sugerencias. Si no hay, se queda el del RMS.
        nuevo_inicio = int(np.median(candidatos_ini)) if candidatos_ini.size > 0 else ini_rms

        # Para el FIN:
        # Buscamos todas las sugerencias de fin que estén cerca del fin del RMS.
        candidatos_fin = fines_sugeridos[(fines_sugeridos >= fin_rms - ventana) & 
                                          (fines_sugeridos <= fin_rms + ventana)]
        # El nuevo fin será la mediana.
        nuevo_fin = int(np.median(candidatos_fin)) if candidatos_fin.size > 0 else fin_rms
        
        if nuevo_fin > nuevo_inicio:
            eventos_refinados.append((nuevo_inicio, nuevo_fin))

    # --- PASO 5: Unir y filtrar como siempre ---
    unidos = unir_intervalos(eventos_refinados, margen_muestras=mpc // 2)
    min_duracion = int(ciclos_minimos * mpc)
    return [(i, f) for i, f in unidos if (f - i) >= min_duracion]

