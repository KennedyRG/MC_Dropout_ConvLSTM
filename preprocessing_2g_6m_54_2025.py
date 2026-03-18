# -*- coding: utf-8 -*-
"""
Procesamiento de datos SST (Sea Surface Temperature) para modelos ConvLSTM
Created on Tue Jan 27 21:43:59 2026
@author: KRGT
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional

######################################################################
######## Constantes de configuración #################################
class Config:
    """Configuración centralizada del procesamiento de datos"""
    TRAIN_LENGTH = 344  # 320  #desde 1854 hasta anio dic 2013 (320 secuencias de 6 meses)
    LEN_YEAR = 344      # Longitud total en años [(2025-1854+1)*12]/6=344
    LEN_SEQ = 6         # Longitud de cada secuencia (6 meses)
    MAP_HEIGHT = 12     # Altura del mapa (región NINO3.4)
    MAP_WIDTH = 75      # Anchura del mapa (región NINO3.4)
    
    # Normalización (calculados del dataset 1854-2025)
    MAX_TEMP = 36.008137           # Valor máximo de SST
    MIN_TEMP = 13.765629           # Valor mínimo de SST
    MEAN_TEMP = 27.125076135105203 # Valor medio de SST
    
    # Región NINO3.4 (5°S-5°N, 170°W-120°W)
    LAT_START = 39  # Índice inicio latitud (5°S)
    LAT_END = 51    # Índice fin latitud (5°N)
    LON_START = 69  # Índice inicio longitud (170°W)
    LON_END = 144   # Índice fin longitud (120°W)
    
    # Ruta de datos
    DATA_PATH = Path('/tf/workspace/test2g_12_2025.npz')


#######################################################################
################## FUNCIONES DE NORMALIZACIÓN #########################
#######################################################################

def normalization(data: np.ndarray) -> np.ndarray:
    """
    Normaliza datos SST al rango [0, 1] usando Min-Max scaling
    
    Args:
        data: Array 2D de temperaturas
        
    Returns:
        Array normalizado en rango [0, 1]
    """
    return (data - Config.MIN_TEMP) / (Config.MAX_TEMP - Config.MIN_TEMP)


def inverse_normalization(data: np.ndarray) -> np.ndarray:
    """
    Revierte la normalización de datos SST
    
    Args:
        data: Array normalizado [0, 1]
        
    Returns:
        Array con valores originales en grados Celsius
    """
    return data * (Config.MAX_TEMP - Config.MIN_TEMP) + Config.MIN_TEMP


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula el Mean Absolute Percentage Error (MAPE)
    
    Args:
        y_true: Valores reales
        y_pred: Valores predichos
        
    Returns:
        MAPE como porcentaje (0-100)
    """
    # Evitar división por cero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


#######################################################################
################## FUNCIONES DE CARGA DE DATOS ########################
#######################################################################

def load_sst_data(data_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], 
                                             Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Carga datos SST desde archivo NPZ con soporte para MaskedArray
    
    Args:
        data_path: Ruta al archivo NPZ
        
    Returns:
        Tupla (sst_data, time, lat, lon)
        - sst_data: Array 3D con datos SST (time, lat, lon)
        - time: Array 1D con timestamps (opcional)
        - lat: Array 1D con latitudes (opcional)
        - lon: Array 1D con longitudes (opcional)
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no contiene datos válidos
    """
    if not data_path.exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo: {data_path}")
    
    print(f"📂 Cargando archivo: {data_path.name}")
    
    try:
        with np.load(data_path, allow_pickle=True) as npz:
            keys = list(npz.keys())
            print(f"   Claves disponibles: {keys}")
            
            # Cargar datos y máscara (formato recomendado)
            if 'data' in keys and 'mask' in keys:
                print("   ✓ Detectado formato MaskedArray (data + mask)")
                data = npz['data']
                mask = npz['mask']
                
                # Reconstruir MaskedArray
                sst_masked = np.ma.MaskedArray(data=data, mask=mask)
                
                # Estadísticas de valores enmascarados
                n_masked = np.ma.count_masked(sst_masked)
                total = sst_masked.size
                pct_masked = (n_masked / total) * 100
                print(f"   Valores enmascarados: {n_masked:,} ({pct_masked:.2f}%)")
                
                # Rellenar valores enmascarados
                # Opción 1: Con la media (mejor para ML)
                #fill_value = np.ma.mean(sst_masked)
                #sst_data = sst_masked.filled(fill_value)
                #print(f"   Rellenado con media: {fill_value:.2f}°C")
                
                # Opción 2: Con NaN (descomentar si prefieres esto) esto para prediccion
                sst_data = sst_masked.filled(np.nan)
                print("   Rellenado con NaN")
            
            # Formato alternativo (clave directa)
            elif 'sst_data' in keys:
                print("   ✓ Detectado formato directo (sst_data)")
                sst_data = npz['sst_data']
            
            # Formato numpy estándar
            elif 'arr_0' in keys:
                print("   ⚠️  Usando formato numpy estándar (arr_0)")
                sst_data = npz['arr_0']
            
            else:
                raise ValueError(f"No se encontró clave válida. Claves: {keys}")
            
            # Cargar metadatos opcionales
            time = npz['time'] if 'time' in keys else None
            lat = npz['lat'] if 'lat' in keys else None
            lon = npz['lon'] if 'lon' in keys else None
            
            # Validar shape
            if sst_data.ndim != 3:
                raise ValueError(
                    f"Se esperaba array 3D (time, lat, lon), "
                    f"se obtuvo shape {sst_data.shape}"
                )
            
            # Convertir a float64
            sst_data = np.array(sst_data, dtype=np.float64)
            
            # Información del dataset
            print(f"   Shape: {sst_data.shape} (time={sst_data.shape[0]}, "
                  f"lat={sst_data.shape[1]}, lon={sst_data.shape[2]})")
            print(f"   Rango: [{np.nanmin(sst_data):.2f}, {np.nanmax(sst_data):.2f}]°C")
            print(f"   Media: {np.nanmean(sst_data):.2f}°C")
            
            return sst_data, time, lat, lon
            
    except Exception as e:
        raise ValueError(f"❌ Error al cargar datos: {str(e)}")


def extract_nino_region(sst_data: np.ndarray) -> np.ndarray:
    """
    Extrae la región NINO3.4 de los datos completos
    Región: 5°S-5°N, 170°W-120°W
    
    Args:
        sst_data: Array completo de datos SST (time, lat, lon)
        
    Returns:
        Array con solo la región NINO3.4 (time, 12, 75)
    """
    nino_data = sst_data[:, Config.LAT_START:Config.LAT_END, 
                         Config.LON_START:Config.LON_END]
    
    print(f"🌍 Región NINO extraída: {nino_data.shape}")
    print(f"   Latitud: índices {Config.LAT_START}-{Config.LAT_END}")
    print(f"   Longitud: índices {Config.LON_START}-{Config.LON_END}")
    
    return nino_data


#######################################################################
############### FUNCIONES DE PREPARACIÓN DE SECUENCIAS ###############
#######################################################################

def create_sequences(sst_data: np.ndarray, length: int) -> np.ndarray:
    """
    Crea secuencias normalizadas para entrenamiento
    Cada secuencia contiene LEN_SEQ (6) timesteps consecutivos
    
    Args:
        sst_data: Datos SST de la región NINO3.4 (time, lat, lon)
        length: Número de secuencias a crear
        
    Returns:
        Array 5D con forma (length, len_seq, height, width, channels)
        
    Raises:
        ValueError: Si no hay suficientes datos para crear las secuencias
    """
    required_timesteps = length * Config.LEN_SEQ
    available_timesteps = sst_data.shape[0]
    
    if required_timesteps > available_timesteps:
        raise ValueError(
            f"Datos insuficientes: se necesitan {required_timesteps} timesteps "
            f"para crear {length} secuencias, pero solo hay {available_timesteps}"
        )
    
    sequences = np.zeros(
        (length, Config.LEN_SEQ, Config.MAP_HEIGHT, Config.MAP_WIDTH, 1), 
        dtype=np.float64
    )
    
    print(f"📊 Creando {length} secuencias de {Config.LEN_SEQ} meses...")
    
    for i in range(length):
        for k in range(Config.LEN_SEQ):
            idx = i * Config.LEN_SEQ + k
            sequences[i, k, :, :, 0] = normalization(sst_data[idx])
    
    print(f"   ✓ Secuencias creadas: {sequences.shape}")
    
    return sequences


def create_target_sequences(train_X: np.ndarray) -> np.ndarray:
    """
    Crea secuencias objetivo (shifted por 1 timestep)
    
    Estrategia:
    - Para timesteps 0-4 de cada secuencia: usar el siguiente timestep
    - Para timestep 5 de cada secuencia: usar el timestep 0 de la siguiente
    - Para la última secuencia: repetir el último timestep
    
    Args:
        train_X: Secuencias de entrada (n_samples, len_seq, height, width, channels)
        
    Returns:
        Secuencias objetivo con mismo shape que train_X
    """
    train_length = train_X.shape[0]
    train_Y = np.zeros_like(train_X)
    
    print(f"🎯 Creando secuencias objetivo...")
    
    for i in range(train_length):
        for k in range(Config.LEN_SEQ):
            if k < Config.LEN_SEQ - 1:
                # Usar el siguiente timestep dentro de la misma secuencia
                train_Y[i, k] = train_X[i, k + 1]
            elif i < train_length - 1:
                # Usar el primer timestep de la siguiente secuencia
                train_Y[i, k] = train_X[i + 1, 0]
            else:
                # Última secuencia: usar el mismo timestep
                train_Y[i, k] = train_X[i, k]
    
    print(f"   ✓ Secuencias objetivo creadas: {train_Y.shape}")
    
    return train_Y


#######################################################################
################# FUNCIÓN PRINCIPAL DE CARGA ##########################
#######################################################################

def load_data_convlstm_monthly(
    train_length: int = Config.TRAIN_LENGTH,
    data_path: Path = Config.DATA_PATH
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga y preprocesa datos SST mensuales para ConvLSTM
    
    Pipeline completo:
    1. Cargar datos desde NPZ
    2. Extraer región NINO3.4
    3. Crear secuencias normalizadas
    4. Crear secuencias objetivo (shifted)
    
    Args:
        train_length: Número de secuencias de entrenamiento (default: 344)
        data_path: Ruta al archivo NPZ de datos
        
    Returns:
        Tupla (normalized_sst, train_X, train_Y)
        - normalized_sst: Dataset completo normalizado
        - train_X: Secuencias de entrenamiento (entrada)
        - train_Y: Secuencias objetivo (salida esperada)
        
    Example:
        >>> normalized_sst, train_X, train_Y = load_data_convlstm_monthly()
        >>> print(f"Training data: {train_X.shape}")
    """
    print("=" * 70)
    print("🚀 INICIANDO CARGA Y PREPROCESAMIENTO DE DATOS SST")
    print("=" * 70)
    
    # 1. Cargar datos
    sst_data, time, lat, lon = load_sst_data(data_path)
    print(f"\n📈 Dataset original: {sst_data.shape}")
    
    # 2. Extraer región NINO3.4
    print("\n" + "=" * 70)
    sst_nino = extract_nino_region(sst_data)
    
    # 3. Mostrar estadísticas de la región
    print("\n📊 ESTADÍSTICAS REGIÓN NINO3.4:")
    print("=" * 70)
    print(f"   Min:  {np.nanmin(sst_nino):.2f}°C")
    print(f"   Max:  {np.nanmax(sst_nino):.2f}°C")
    print(f"   Mean: {np.nanmean(sst_nino):.6f}°C")
    print(f"   Std:  {np.nanstd(sst_nino):.2f}°C")
    
    # 4. Crear secuencias normalizadas
    print("\n" + "=" * 70)
    normalized_sst = create_sequences(sst_nino, Config.LEN_YEAR)
    train_X = normalized_sst[:train_length]
    
    # 5. Crear secuencias objetivo
    print()
    train_Y = create_target_sequences(train_X)
    
    # 6. Resumen final
    print("\n" + "=" * 70)
    print("✅ DATOS CARGADOS EXITOSAMENTE")
    print("=" * 70)
    print(f"   Dataset completo:     {normalized_sst.shape}")
    print(f"   Secuencias entrada:   {train_X.shape}")
    print(f"   Secuencias objetivo:  {train_Y.shape}")
    print(f"   Periodo temporal:     {Config.LEN_YEAR * Config.LEN_SEQ} meses")
    print(f"   Secuencias training:  {train_length}")
    print("=" * 70 + "\n")
    
    return normalized_sst, train_X, train_Y


#######################################################################
##################### EJECUCIÓN PRINCIPAL #############################
#######################################################################

if __name__ == "__main__":
    # Cargar datos
    normalized_sst, train_X_raw, train_Y_raw = load_data_convlstm_monthly()
    
    # Validación básica
    print("🔍 VALIDACIÓN DE DATOS:")
    print("-" * 70)
    print(f"   Rango train_X: [{train_X_raw.min():.4f}, {train_X_raw.max():.4f}]")
    print(f"   Rango train_Y: [{train_Y_raw.min():.4f}, {train_Y_raw.max():.4f}]")
    print(f"   NaN en train_X: {np.isnan(train_X_raw).sum()}")
    print(f"   NaN en train_Y: {np.isnan(train_Y_raw).sum()}")
    print("-" * 70)