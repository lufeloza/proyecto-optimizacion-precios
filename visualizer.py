import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib

# Configuración no interactiva para evitar bloqueos en ejecución automática
matplotlib.use('Agg')

def graficar_optimizacion_t7(df: pd.DataFrame, mejor_precio: float):
    """Crea la gráfica básica 2D para T-7."""
    plt.figure(figsize=(10, 6))
    plt.plot(df['Precio'], df['Ingreso_Esperado'], color='#4338ca', linewidth=3, label='Ingreso Esperado')
    plt.scatter(mejor_precio, df['Ingreso_Esperado'].max(), color='#f59e0b', s=100, zorder=5, label=f'Óptimo (${mejor_precio:,.0f})')
    plt.title('Optimización de Ingresos para T-7', fontsize=14, fontweight='bold')
    plt.xlabel('Precio ($)')
    plt.ylabel('Ingreso Esperado ($)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig('optimizacion_T7.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfica 2D generada.")

def graficar_superficie_3d(engine):
    """Genera superficie 3D: Precio vs Días vs Ingreso."""
    precios = np.linspace(30000, 100000, 50)
    dias = np.linspace(1, 30, 30)
    P, D = np.meshgrid(precios, dias)
    
    # Crear DataFrame para predicción masiva
    flat_pd = pd.DataFrame({'Precio': P.flatten(), 'Dias_Faltantes': D.flatten()})
    flat_pd['Probabilidad'] = engine.modelo.predict_proba(flat_pd[['Precio', 'Dias_Faltantes']])[:, 1]
    flat_pd['Ingreso_Esperado'] = flat_pd['Precio'] * flat_pd['Probabilidad']
    
    I = flat_pd['Ingreso_Esperado'].values.reshape(P.shape)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(P, D, I, cmap='viridis', edgecolor='none', alpha=0.8)
    
    ax.set_title('Mapa de Calor 3D: Optimización Global', fontsize=15, pad=20)
    ax.set_xlabel('Precio ($)', fontsize=12)
    ax.set_ylabel('Días Faltantes', fontsize=12)
    ax.set_zlabel('Ingreso Esperado ($)', fontsize=12)
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='Ingreso Esperado')
    plt.savefig('heatmap_3d.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfica 3D generada.")

def graficar_evolucion_precios(engine, dias_lista=[25, 15, 7, 2]):
    """Compara la curva de optimización en diferentes puntos del tiempo."""
    plt.figure(figsize=(12, 7))
    colores = ['#94a3b8', '#6366f1', '#4338ca', '#1e1b4b']
    
    for i, d in enumerate(dias_lista):
        curva = engine.obtener_curva_optimizacion(d)
        mejor = curva.loc[curva['Ingreso_Esperado'].idxmax()]
        plt.plot(curva['Precio'], curva['Ingreso_Esperado'], label=f'T-{d} días', color=colores[i], linewidth=2.5)
        plt.scatter(mejor['Precio'], mejor['Ingreso_Esperado'], color=colores[i], s=80, edgecolors='white')

    plt.title('Evolución Dinámica: Curva de Optimización según el Tiempo', fontsize=14, fontweight='bold')
    plt.xlabel('Precio ($)')
    plt.ylabel('Ingreso Esperado ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('evolucion_temporal.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Gráfica de evolución generada.")
