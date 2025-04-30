import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chardet

# Configuración inicial
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
COLOR_PALETTE = "viridis"
TAMAÑOS = ['Pequeño', 'Mediano', 'Grande']

def detectar_encoding(archivo):
    """Detecta la codificación del archivo"""
    with open(archivo, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def cargar_csv(nombre_archivo):
    """Carga y procesa un archivo CSV de resultados"""
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"No se encuentra el archivo {nombre_archivo}")
    
    encoding = detectar_encoding(nombre_archivo)
    df = pd.read_csv(nombre_archivo, encoding=encoding)
    return df.melt(id_vars=['Algoritmo', 'Tamaño'],
                 value_vars=['Mejor', 'Peor', 'Media', 'Desviacion'],
                 var_name='Metrica', 
                 value_name='Valor')

def generar_grafico_por_tamaño(df, titulo_base, ylabel, unidad='', tamaño=''):
    """Genera gráficos separados por tamaño de parámetros"""
    df_filtrado = df[df['Tamaño'] == tamaño]
    
    if df_filtrado.empty:
        print(f"No hay datos para el tamaño {tamaño}")
        return
    
    # Ordenar por mejor valor
    mejor_por_algoritmo = df_filtrado[df_filtrado['Metrica'] == 'Mejor'].groupby('Algoritmo')['Valor'].min()
    orden = mejor_por_algoritmo.sort_values().index.tolist()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{titulo_base} - Tamaño {tamaño}", fontsize=16)
    
    metricas = ['Mejor', 'Peor', 'Media', 'Desviacion']
    for i, metrica in enumerate(metricas):
        ax = axs[i//2, i%2]
        sns.barplot(data=df_filtrado[df_filtrado['Metrica'] == metrica], 
                            x='Algoritmo', 
                            y='Valor', 
                            order=orden,
                            hue='Algoritmo',  # <- Añade esto
                            palette=COLOR_PALETTE,
                            legend=False,     # <- Y esto
                            ax=ax)
        ax.set_title(f'{metrica}')
        ax.set_xlabel('')
        ax.set_ylabel(ylabel if i%2 == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        
        # Añadir valores
        for p in ax.patches:
            texto = f"{p.get_height():.5f}{unidad}" if unidad else f"{p.get_height():.5f}"
            ax.annotate(texto, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=8)
    
    plt.tight_layout()
    return fig

def generar_graficas_metricas():
    """Genera todas las gráficas separadas por tamaño"""
    try:
        df_calidad = cargar_csv('calidad_solucion.csv')
        df_tiempo = cargar_csv('tiempo_ejecucion.csv')
        df_memoria = cargar_csv('uso_memoria.csv')
    except FileNotFoundError as e:
        print(f"Error: {str(e)} - Ejecuta primero comparar_algoritmos.py")
        return

    # Generar gráficas para cada tamaño
    for tamaño in TAMAÑOS:
        # Calidad de solución
        fig = generar_grafico_por_tamaño(df_calidad, 'Calidad de Solución', 'Fitness', tamaño=tamaño)
        fig.savefig(f'calidad_solucion_{tamaño.lower()}.png')
        plt.close()
        
        # Tiempos de ejecución
        fig = generar_grafico_por_tamaño(df_tiempo, 'Tiempos de Ejecución', 'Segundos', 's', tamaño)
        for ax in fig.axes:
            ax.set_yscale("log")
        fig.savefig(f'tiempos_ejecucion_{tamaño.lower()}.png')
        plt.close()
        
        # Uso de memoria
        fig = generar_grafico_por_tamaño(df_memoria, 'Uso de Memoria', 'MB', ' MB', tamaño)
        fig.savefig(f'uso_memoria_{tamaño.lower()}.png')
        plt.close()

def generar_reporte_completo():
    """Genera el reporte HTML con gráficas separadas por tamaño"""
    generar_graficas_metricas()
    
    # Crear reporte HTML
    with open('reporte_comparacion.html', 'w', encoding='utf-8') as f:
        f.write("""<html>
<head>
    <title>Reporte de Comparación por Tamaño</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        h1, h2, h3 { color: #2c3e50; }
        img { max-width: 100%; margin: 1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .container { max-width: 1200px; margin: 0 auto; }
        .tamaño-section { margin-bottom: 3rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Reporte Comparativo por Tamaño de Parámetros</h1>""")

        for tamaño in TAMAÑOS:
            f.write(f"""
        <div class="tamaño-section">
            <h2>Tamaño {tamaño}</h2>
            
            <h3>Calidad de Solución</h3>
            <img src="calidad_solucion_{tamaño.lower()}.png" alt="Calidad {tamaño}">
            
            <h3>Tiempos de Ejecución (escala logarítmica)</h3>
            <img src="tiempos_ejecucion_{tamaño.lower()}.png" alt="Tiempos {tamaño}">
            
            <h3>Uso de Memoria</h3>
            <img src="uso_memoria_{tamaño.lower()}.png" alt="Memoria {tamaño}">
        </div>""")

        f.write("""
    </div>
</body>
</html>""")

if __name__ == "__main__":
    generar_reporte_completo()
    print("Reporte generado: reporte_comparacion.html")