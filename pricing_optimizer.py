import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict

class PricingEngine:
    """
    Motor de optimización de precios dinámicos basado en Regresión Logística
    y Esperanza Matemática del ingreso.
    """

    def __init__(self):
        self.modelo = LogisticRegression()
        self.is_trained = False

    def generar_datos_sinteticos(self, n: int = 1000) -> pd.DataFrame:
        """Genera datos de ventas simulados basados en lógica de mercado."""
        np.random.seed(42)
        precios = np.random.uniform(30000, 100000, n)
        dias = np.random.randint(1, 30, n)
        
        # Lógica: mayor precio = menor probabilidad / más días = mayor probabilidad
        # Representa la curva de demanda (descendente al subir precio)
        z = 2.0 - 0.00007 * precios + 0.1 * dias
        prob_venta = 1 / (1 + np.exp(-z))
        vendido = np.random.binomial(1, prob_venta)
        
        return pd.DataFrame({'Precio': precios, 'Dias_Faltantes': dias, 'Vendido': vendido})

    def entrenar(self, df: pd.DataFrame) -> Dict[str, float]:
        """Entrena el modelo y retorna métricas de desempeño."""
        X = df[['Precio', 'Dias_Faltantes']]
        y = df['Vendido']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.modelo.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluación profesional
        preds = self.modelo.predict(X_test)
        probs = self.modelo.predict_proba(X_test)[:, 1]
        
        metricas = {
            'accuracy': accuracy_score(y_test, preds),
            'roc_auc': roc_auc_score(y_test, probs)
        }
        return metricas

    def obtener_curva_optimizacion(self, dias_restantes: int) -> pd.DataFrame:
        """Genera el escenario para una curva de optimización completa."""
        precios_test = np.arange(30000, 105000, 1000)
        escenarios = pd.DataFrame({
            'Precio': precios_test, 
            'Dias_Faltantes': dias_restantes
        })
        escenarios['Probabilidad'] = self.modelo.predict_proba(escenarios[['Precio', 'Dias_Faltantes']])[:, 1]
        escenarios['Ingreso_Esperado'] = escenarios['Precio'] * escenarios['Probabilidad']
        return escenarios

    def optimizar_precio(self, dias_restantes: int) -> pd.Series:
        """Encuentra el precio que maximiza el ingreso esperado."""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado antes de optimizar.")
            
        curva = self.obtener_curva_optimizacion(dias_restantes)
        return curva.loc[curva['Ingreso_Esperado'].idxmax()]

if __name__ == "__main__":
    # Inicialización del motor
    engine = PricingEngine()
    
    # Flujo de datos
    print("--- 🧠 ENTRENAMIENTO DEL MOTOR ---")
    datos = engine.generar_datos_sinteticos()
    metricas = engine.entrenar(datos)
    
    print(f"Modelo entrenado con precisión: {metricas['accuracy']:.2%}")
    print(f"ROC AUC Score: {metricas['roc_auc']:.4f}")
    
    # Ejemplo de optimización
    dias_consulta = 7
    mejor_escenario = engine.optimizar_precio(dias_consulta)
    
    print(f"\n--- 🚀 RESULTADO PARA T-{dias_consulta} DÍAS ---")
    print(f"Sugerencia de Precio: ${mejor_escenario['Precio']:,.0f}")
    print(f"Probabilidad de Venta: {mejor_escenario['Probabilidad']:.1%}")
    print(f"Ingreso Máximo Esperado: ${mejor_escenario['Ingreso_Esperado']:,.0f}")

    # --- Generación de visualización avanzada (Wow Factor) ---
    try:
        from visualizer import graficar_optimizacion_t7, graficar_superficie_3d, graficar_evolucion_precios
        
        # 1. Gráfica 2D estándar
        curva_t7 = engine.obtener_curva_optimizacion(dias_consulta)
        graficar_optimizacion_t7(curva_t7, mejor_escenario['Precio'])
        
        # 2. Mapa de Calor 3D (Vista global del negocio)
        graficar_superficie_3d(engine)
        
        # 3. Evolución Dinámica (T-25 a T-2)
        graficar_evolucion_precios(engine)
        
        print("\n✨ ¡Toque 'Wow' completado! Las 3 gráficas premium están listas.")
        
    except ImportError as e:
        print(f"\n⚠️ Error al importar visualizer: {e}")