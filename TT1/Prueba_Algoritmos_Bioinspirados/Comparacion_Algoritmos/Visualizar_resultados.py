import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import chardet

# Configuración actualizada
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
COLOR_PALETTE = "viridis"
EVALUACIONES = ['100,000 evaluaciones', '500,000 evaluaciones', '2,000,000 evaluaciones']

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

def cargar_estabilidad(nombre_archivo):
    """Carga el archivo de estabilidad"""
    if not os.path.exists(nombre_archivo):
        raise FileNotFoundError(f"No se encuentra el archivo {nombre_archivo}")
    
    encoding = detectar_encoding(nombre_archivo)
    return pd.read_csv(nombre_archivo, encoding=encoding)

def generar_grafico_por_tamaño(df, titulo_base, ylabel, unidad='', evaluaciones=''):
    """Genera gráficos separados por número de evaluaciones"""
    df_filtrado = df[df['Tamaño'] == evaluaciones]
    
    if df_filtrado.empty:
        print(f"No hay datos para {evaluaciones}")
        return
    
    # Ordenar por mejor valor
    mejor_por_algoritmo = df_filtrado[df_filtrado['Metrica'] == 'Mejor'].groupby('Algoritmo')['Valor'].min()
    orden = mejor_por_algoritmo.sort_values().index.tolist()
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"{titulo_base} - {evaluaciones}", fontsize=16)
    
    metricas = ['Mejor', 'Peor', 'Media', 'Desviacion']
    for i, metrica in enumerate(metricas):
        ax = axs[i//2, i%2]
        sns.barplot(data=df_filtrado[df_filtrado['Metrica'] == metrica], 
                    x='Algoritmo', 
                    y='Valor', 
                    hue='Algoritmo',
                    order=orden,
                    palette=COLOR_PALETTE,
                    legend=False,
                    ax=ax)
        ax.set_title(f'{metrica}')
        ax.set_xlabel('')
        ax.set_ylabel(ylabel if i%2 == 0 else '')
        ax.tick_params(axis='x', rotation=45)
        
        # Añadir valores
        for p in ax.patches:
            texto = f"{p.get_height():.2f}{unidad}" if unidad else f"{p.get_height():.2f}"
            ax.annotate(texto, 
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        fontsize=8)
    
    plt.tight_layout()
    return fig

def generar_grafico_estabilidad(df, evaluaciones):
    """Genera gráfico de estabilidad ordenado de mejor a peor"""
    df_filtrado = df[df['Tamaño'] == evaluaciones]
    
    if df_filtrado.empty:
        print(f"No hay datos para {evaluaciones}")
        return None
    
    # Ordenar de menor a mayor coeficiente de variación
    orden = df_filtrado.sort_values('CoeficienteVariacion')['Algoritmo'].tolist()
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_filtrado, 
                     x='Algoritmo', 
                     y='CoeficienteVariacion', 
                     hue='Algoritmo',
                     order=orden,
                     palette=COLOR_PALETTE,
                     legend=False)
    plt.title(f'Estabilidad - {evaluaciones}')
    plt.ylabel('Coeficiente de Variación (%)')
    plt.xlabel('Algoritmo')
    plt.xticks(rotation=45)
    
    # Añadir etiquetas
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', 
                    xytext=(0, 5), 
                    textcoords='offset points',
                    fontsize=8)
    plt.tight_layout()
    return plt.gcf()

def generar_graficas_metricas():
    """Genera todas las gráficas separadas por evaluaciones"""
    try:
        df_calidad = cargar_csv('calidad_solucion.csv')
        df_tiempo = cargar_csv('tiempo_ejecucion.csv')
        df_estabilidad = cargar_estabilidad('estabilidad.csv')
    except FileNotFoundError as e:
        print(f"Error: {str(e)} - Ejecuta primero comparar_algoritmos.py")
        return

    # Generar gráficas para cada evaluación
    for evaluaciones in EVALUACIONES:
        # Calidad de solución
        fig = generar_grafico_por_tamaño(df_calidad, 'Calidad de Solución', 'Fitness', evaluaciones=evaluaciones)
        fig.savefig(f'calidad_solucion_{evaluaciones.lower().replace(",", "").replace(" ", "_")}.png')
        plt.close()
        
        # Tiempos de ejecución
        fig = generar_grafico_por_tamaño(df_tiempo, 'Tiempos de Ejecución', 'Segundos', 's', evaluaciones)
        for ax in fig.axes:
            ax.set_yscale("log")
        fig.savefig(f'tiempos_ejecucion_{evaluaciones.lower().replace(",", "").replace(" ", "_")}.png')
        plt.close()
        
        # Estabilidad
        fig = generar_grafico_estabilidad(df_estabilidad, evaluaciones)
        if fig:
            fig.savefig(f'estabilidad_{evaluaciones.lower().replace(",", "").replace(" ", "_")}.png')
            plt.close()

def generar_reporte_completo():
    """Genera el reporte HTML con las nuevas etiquetas"""
    generar_graficas_metricas()
    
    # Crear reporte HTML
    with open('reporte_comparacion.html', 'w', encoding='utf-8') as f:
        f.write("""<html>
<head>
    <title>Reporte de Comparación por Evaluaciones</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 2em; }
        h1, h2, h3 { color: #2c3e50; }
        img { max-width: 100%; margin: 1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .container { max-width: 1200px; margin: 0 auto; }
        .evaluacion-section { margin-bottom: 3rem; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Reporte Comparativo por Número de Evaluaciones</h1>""")

        for evaluaciones in EVALUACIONES:
            filename = evaluaciones.lower().replace(",", "").replace(" ", "_")
            f.write(f"""
        <div class="evaluacion-section">
            <h2>{evaluaciones}</h2>
            
            <h3>Calidad de Solución</h3>
            <img src="calidad_solucion_{filename}.png" alt="Calidad {evaluaciones}">
            
            <h3>Tiempos de Ejecución (escala logarítmica)</h3>
            <img src="tiempos_ejecucion_{filename}.png" alt="Tiempos {evaluaciones}">
            
            <h3>Estabilidad</h3>
            <img src="estabilidad_{filename}.png" alt="Estabilidad {evaluaciones}">
        </div>""")

        f.write("""
    </div>
</body>
</html>""")

if __name__ == "__main__":
    generar_reporte_completo()
    print("Reporte generado: reporte_comparacion.html")