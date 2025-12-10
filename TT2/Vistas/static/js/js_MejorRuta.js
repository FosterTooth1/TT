document.addEventListener("DOMContentLoaded", function() {
    // 1. Inicialización de Mapas
    // Mapa 1: Penalizado
    const mapPenalizado = L.map('mapa-penalizado', { minZoom: 5 }).setView([19.35, -99.75], 8);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
    }).addTo(mapPenalizado);

    // Mapa 2: Limpio
    const mapLimpio = L.map('mapa-limpio', { minZoom: 5 }).setView([19.35, -99.75], 8);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
    }).addTo(mapLimpio);

    // 2. Definir y aplicar límites geográficos
    const northEast = L.latLng(20.35, -98.5);
    const southWest = L.latLng(18.5, -100.5);
    const bounds = L.latLngBounds(southWest, northEast);
    mapPenalizado.setMaxBounds(bounds);
    mapLimpio.setMaxBounds(bounds);

    // Añadir capa base del mapa (el fondo)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; CARTO',
    }).addTo(map);
    
    // 3. Definir iconos personalizados
    // Icono Verde
    const iconoVerde = new L.Icon({
        iconUrl: '/static/imagenes/marker-icon-2x-green.png',
        shadowUrl: '/static/imagenes/marker-shadow.png',
        iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
    });
    
    // Icono Azul
    const iconoAzul = new L.Icon.Default(); 

    // Icono Rojo
    const iconoRojo = new L.Icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
    });


    // 4. Procesar Datos
    const rutaData = sessionStorage.getItem('rutaOptimizada');
    
    if (!rutaData) {
        alert("No se encontraron datos de ruta. Regresando...");
        window.location.href = '/';
        return;
    }

    try {
        const data = JSON.parse(rutaData);
        console.log("Datos recibidos:", data);

        // Actualizar Tabla de Estadísticas
        // Fitness Penalizado
        if (document.getElementById('fitness-valor')) {
            document.getElementById('fitness-valor').textContent = data.fitness_penalizado ? data.fitness_penalizado.toFixed(2) : "N/A";
        }
        // Distancia Real
        if (document.getElementById('distancia-real-valor')) {
            document.getElementById('distancia-real-valor').textContent = data.distancia_real ? data.distancia_real.toFixed(2) : "N/A";
        }

        // Moda del Clima (Usando ruta penalizada)
        const climaModaEl = document.getElementById('clima-moda');
        if (climaModaEl && data.ruta_penalizada) {
            const condiciones = data.ruta_penalizada.map(p => p.condicion).filter(c => c);
            if (condiciones.length > 0) {
                const conteo = condiciones.reduce((acc, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
                const moda = Object.keys(conteo).reduce((a, b) => conteo[a] > conteo[b] ? a : b);
                climaModaEl.textContent = moda;
            } else {
                climaModaEl.textContent = "N/A";
            }
        }

        // Renderizar Mapa 1: Penalizado
        const markersPenalizados = [];
        const tabla = document.getElementById('tabla-mejor-ruta');
        
        data.ruta_penalizada.forEach((punto, i) => {
            const latLng = [punto.lat, punto.lng];
            const esInicio = (i === 0);
            
            // Marcador
            const marker = L.marker(latLng, { icon: esInicio ? iconoVerde : iconoAzul })
                .addTo(mapPenalizado)
                .bindPopup(`<b>${i + 1}. ${punto.nombre}</b><br>${punto.condicion}`);
            
            markersPenalizados.push(marker);

            // Tabla
            const row = tabla.insertRow();
            row.insertCell(0).textContent = i + 1;
            row.insertCell(1).textContent = punto.nombre;
            row.insertCell(2).textContent = punto.condicion;
        });

        if (markersPenalizados.length > 0) {
            const grupo = L.featureGroup(markersPenalizados);
            mapPenalizado.fitBounds(grupo.getBounds().pad(0.1));
            dibujarRuta(data.ruta_penalizada, mapPenalizado, '#08564d'); // Color verde oscuro corporativo
        }

        // Renderizar Mapa 2: Limpio
        // Aquí se compara el orden con la ruta penalizada
        const markersLimpios = [];
        
        if (data.ruta_limpia) {
            data.ruta_limpia.forEach((punto, i) => {
                const latLng = [punto.lat, punto.lng];
                
                // Lógica de color:
                // Si es el inicio (i=0), Verde.
                // Si el nombre en esta posición (i) es diferente al nombre en la misma posición de ruta_penalizada, es Rojo.
                // Si es igual, Azul.
                
                let iconToUse = iconoAzul;
                let esDiferente = false;

                if (i === 0) {
                    iconToUse = iconoVerde;
                } else {
                    const nombreEnPenalizada = data.ruta_penalizada[i] ? data.ruta_penalizada[i].nombre : null;
                    if (nombreEnPenalizada !== punto.nombre) {
                        iconToUse = iconoRojo;
                        esDiferente = true;
                    }
                }

                const msgComparativa = esDiferente ? "<br><span style='color:red'>⚠️ Posición diferente</span>" : "";

                const marker = L.marker(latLng, { icon: iconToUse })
                    .addTo(mapLimpio)
                    .bindPopup(`<b>${i + 1}. ${punto.nombre}</b>${msgComparativa}`);
                
                markersLimpios.push(marker);
            });

            if (markersLimpios.length > 0) {
                const grupo = L.featureGroup(markersLimpios);
                mapLimpio.fitBounds(grupo.getBounds().pad(0.1));
                dibujarRuta(data.ruta_limpia, mapLimpio, '#417fc6');
            }
        }

        // Gráfica de Fitness por Generación 
        renderizarGrafica(data.fitness_generaciones);

    } catch (error) {
        console.error("Error JS:", error);
    }

    /* ########################################################################################################################### */ 
    // 5. Función para dibujar la ruta
    function dibujarRuta(coordenadas) {
        // ***** INICIO DEL CAMBIO *****
        // Hacemos una copia para no modificar el array original
        const coordsParaRuta = [...coordenadas]; 
        if (coordsParaRuta.length > 0) {
            // Añadimos el primer punto al final para cerrar el ciclo
            coordsParaRuta.push(coordsParaRuta[0]); 
        }
        // ***** FIN DEL CAMBIO *****

        const urlCoordinates = coordsParaRuta.map(c => `${c.lng},${c.lat}`).join(';');
        const apiUrl = `https://router.project-osrm.org/route/v1/driving/${urlCoordinates}?overview=full&geometries=geojson`;
        
        fetch(apiUrl)
            .then(res => res.json())
            .then(routeData => {
                if (routeData.code !== 'Ok') {
                    throw new Error("No se pudo obtener la ruta desde OSRM.");
                }
                
                const routeGeometry = routeData.routes[0].geometry;
                
                const polyline = L.geoJSON(routeGeometry, {
                    style: {
                        color: 'black',
                        weight: 5,
                        opacity: 0.7
                    }
                }).addTo(map);

            })
            .catch(err => console.error("Error al dibujar la ruta:", err));
    }
    
    /* ########################################################################################################################### */
    // 6. Funcionalidad de los botones
    const guardarRutaBtn = document.getElementById("guardarRuta");
    if (guardarRutaBtn) {
        guardarRutaBtn.addEventListener("click", async () => {
            try {
                const rutaData = sessionStorage.getItem('rutaOptimizada');
                if (!rutaData) {
                    alert("No hay datos de ruta para guardar.");
                    return;
                }

                const data = JSON.parse(rutaData);

                const body = {
                    destinos: data.ruta,
                    indices: data.indices
                };

                const response = await fetch('/guardar-ruta', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });

                const resultado = await response.json();

                if (response.ok) {
                    alert("Ruta guardada correctamente");
                } else {
                    alert("Error al guardar ruta: " + resultado.message);
                }
            } catch (error) {
                console.error("Error al guardar ruta:", error);
                alert("Error al guardar la ruta");
            }
        });
    }
    
    /* ########################################################################################################################### */
    // Botón para generar nueva ruta
    document.getElementById("nuevaRuta").addEventListener("click", () => {
        window.location.href = '/';
    });

    /* ########################################################################################################################### */
    // Manejar cerrar sesión
    const cerrarSesionBtn = document.getElementById('cerrarSesion');
    if (cerrarSesionBtn) {
        cerrarSesionBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/cerrar-sesion', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                if (response.ok) {
                    window.location.reload();
                } else {
                    alert('Error al cerrar sesión');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error al cerrar sesión');
            }
        });
    }
});