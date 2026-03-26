# Power-Quality

**Desarrollo de un Algoritmo de Aprendizaje Automatico para la Clasificacion de Hundimientos de Tension en Microrredes Electricas**

Trabajo de grado -- Programa de Ingenieria Electronica, Universidad de Cundinamarca (Fusagasuga, Colombia, 2025).

- **Autor:** Juan David Saa Lascarra
- **Director:** PhD. Andres Felipe Guerrero Guerrero
- **Co-director:** MsC. Edgar Hernando Criollo Velasquez

---

## Descripcion

Sistema para la deteccion y clasificacion automatica de perturbaciones en senales de Calidad de Potencia (PQ) segun la norma IEEE Std 1159-2019 e IEC 61000-4-30. El proyecto combina:

1. **Detector hibrido por fusion** de tres metodos (RMS, Envolvente de Hilbert y Transformada Wavelet Estacionaria - SWT), que utiliza el RMS como Region de Interes (ROI) y refina los bordes temporales mediante votacion por mediana.
2. **Clasificador basado en Machine Learning** (XGBoost para analisis monofasico, Random Forest para trifasico) entrenado con 24,600 senales sinteticas y validado con 144 registros trifasicos reales del EPRI.

## Estructura del Repositorio

```
Power-Quality/
├── Caracteristicas/
│   ├── algoritmos_deteccion.py         # Implementacion de los 3 detectores y el algoritmo de fusion
│   ├── caracteristicas.ipynb           # Extraccion de caracteristicas monofasicas
│   └── caracteristicas_trifasico.ipynb # Extraccion de caracteristicas trifasicas
│
├── Generacion_Senales/
│   ├── Generacion.ipynb                # Generacion de senales sinteticas monofasicas
│   └── Trifasico/trifasico.ipynb       # Generacion de senales trifasicas (tipos A-G de Bollen)
│
├── Entrenamiento/
│   ├── Entrenamiento.ipynb             # Entrenamiento y validacion cruzada (monofasico)
│   ├── Entrenamiento_trifasico.ipynb   # Entrenamiento y validacion cruzada (trifasico)
│   ├── Mono/                           # Modelo XGBoost monofasico (.joblib)
│   └── Tri/                            # Modelo Random Forest trifasico (.joblib)
│
├── Pruebas_Rendimiento/
│   ├── algoritmos_deteccion.py         # Version del detector para evaluacion de rendimiento
│   └── Pruebas.ipynb                   # Evaluacion con metricas IoU, precision, recall, F1
│
├── Diagramas_Flujo/                    # Diagramas de flujo (.drawio) de cada detector y la fusion
├── imagenes/                           # Graficas de senales, mascaras y resultados (.svg)
│
├── main.ipynb                          # Notebook principal de ejecucion del sistema completo
├── graficar.ipynb                      # Generacion de graficas para el documento
├── unidr_datos.ipynb                   # Union y consolidacion de datasets
│
├── Detector.tex                        # Documentacion tecnica del algoritmo de deteccion
├── Proyecto.tex                        # Documento completo del trabajo de grado
├── referencias.bib                     # Bibliografia en formato BibTeX
└── .gitignore
```

## Algoritmo de Deteccion

El detector opera en tres etapas:

### Etapa 1: Generacion de mascaras individuales

| Detector | Principio | Umbral | Fortaleza |
|---|---|---|---|
| **RMS** | Valor eficaz con ventana de 1 ciclo y 50% solapamiento | Percentil 95 de la secuencia RMS x 0.90 | Robusto ante ruido, define la ROI |
| **Hilbert** | Envolvente instantanea via senal analitica | Percentil 90 de la envolvente x 0.90 (sag) / 1.10 (swell) | Alta resolucion temporal |
| **SWT** | Energia acumulada de coeficientes de detalle (db4, 4 niveles) | Percentil 99 de la energia | Sensible a transitorios y cambios abruptos |

Cada mascara se somete a post-procesamiento morfologico (cierre con M/4 y apertura con M/8 muestras).

### Etapa 2: Fusion ROI + Mediana

1. Las mascaras de Hilbert y SWT se filtran con AND logico contra la mascara RMS (ROI).
2. Se extraen los bordes (inicio/fin) de los detectores auxiliares filtrados.
3. Para cada intervalo del RMS, se buscan candidatos de borde dentro de +-1 ciclo y se toma la **mediana** como estimador robusto.

### Etapa 3: Post-procesamiento

- Union de intervalos separados por menos de medio ciclo.
- Descarte de intervalos con duracion inferior a 0.5 ciclos.

## Clasificacion

### Caracteristicas monofasicas (10 features)
Curtosis, media y desviacion estandar de la envolvente de Hilbert, factor de cresta, energia FFT, profundidad relativa, RMS del segmento, energia/entropia/desviacion de coeficientes wavelet.

### Caracteristicas trifasicas
Componentes simetricas de Fortescue (V0, V1, V2), factor de desbalance (VUF), componentes de Clarke (V_alpha, V_beta).

### Modelos seleccionados

| Modo | Modelo | Accuracy (sintetico) | F1-Score (sintetico) |
|---|---|---|---|
| Monofasico | XGBoost | 98.36% | 98.35% |
| Trifasico | Random Forest | 99.59% | 99.60% |

Validacion cruzada con k=10 folds sobre el 70% de entrenamiento.

## Requisitos

- Python 3.8+
- numpy >= 1.21
- scipy >= 1.7
- PyWavelets >= 1.1
- pandas >= 1.3
- xgboost >= 1.5
- joblib >= 1.1
- matplotlib >= 3.4
- scikit-learn >= 1.0

## Uso rapido

```python
from Caracteristicas.algoritmos_deteccion import detectar_eventos_con_fusion

# senal: array numpy con la senal de tension
# fs: frecuencia de muestreo en Hz
eventos = detectar_eventos_con_fusion(senal, fs)
# Retorna lista de tuplas (muestra_inicio, muestra_fin)
```

## Referencias principales

- IEEE Std 1159-2019 -- Recommended Practice for Monitoring Electric Power Quality
- IEC 61000-4-30 -- Testing and measurement techniques
- Bollen, M.H.J. -- Understanding Power Quality Problems (tipos de fallas A-G)
- Chen & Guestrin (2016) -- XGBoost

## Licencia

Trabajo academico de la Universidad de Cundinamarca. Consultar con el autor para uso fuera del ambito academico.
