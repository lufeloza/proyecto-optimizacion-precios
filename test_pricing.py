import pytest
import pandas as pd
import numpy as np
from pricing_optimizer import PricingEngine

@pytest.fixture
def engine():
    """Fixture para inicializar el motor de precios."""
    return PricingEngine()

def test_generar_datos(engine):
    """Verifica que la generación de datos sea correcta."""
    n = 500
    df = engine.generar_datos_sinteticos(n=n)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == n
    assert list(df.columns) == ['Precio', 'Dias_Faltantes', 'Vendido']
    assert df['Precio'].min() >= 30000
    assert df['Precio'].max() <= 100000

def test_entrenamiento(engine):
    """Verifica que el modelo se entrene y devuelva métricas válidas."""
    df = engine.generar_datos_sinteticos(n=200)
    metricas = engine.entrenar(df)
    
    assert engine.is_trained is True
    assert 'accuracy' in metricas
    assert 'roc_auc' in metricas
    assert 0 <= metricas['accuracy'] <= 1
    assert 0 <= metricas['roc_auc'] <= 1

def test_optimizacion_coherencia(engine):
    """Verifica que la optimización devuelva resultados lógicos."""
    df = engine.generar_datos_sinteticos(n=200)
    engine.entrenar(df)
    
    dias = 7
    mejor = engine.optimizar_precio(dias)
    
    assert isinstance(mejor, pd.Series)
    assert 30000 <= mejor['Precio'] <= 105000
    assert mejor['Ingreso_Esperado'] > 0
    assert 0 <= mejor['Probabilidad'] <= 1

def test_error_si_no_entrenado(engine):
    """Verifica que el sistema falle si se intenta optimizar sin entrenar."""
    with pytest.raises(ValueError, match="El modelo debe ser entrenado"):
        engine.optimizar_precio(10)
