document.addEventListener("DOMContentLoaded", function() {
    
    // 1. Inicializar el mapa
    const map = L.map('mapa', {
        minZoom: 8,
        maxZoom: 14
    }).setView([19.35, -99.75], 8);

    // 2. Definir y aplicar límites geográficos
    const southWest = L.latLng(18.5, -100.5);
    const northEast = L.latLng(20.2, -98.5);
    const bounds = L.latLngBounds(southWest, northEast);
    map.setMaxBounds(bounds);
    map.on('drag', () => map.panInsideBounds(bounds, { animate: true }));

    // 3. Añadir capa base del mapa (el fondo)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; CARTO',
    }).addTo(map);

    // 4. Obtener datos del sessionStorage
    const rutaData = sessionStorage.getItem('rutaOptimizada');
    
    if (!rutaData) {
        console.error("No se encontraron datos de ruta");
        alert("No se encontraron datos de ruta. Regresando a la selección de naves.");
        window.location.href = '/';
        return;
    }

    try {
        const data = JSON.parse(rutaData);
        console.log("Datos de ruta cargados:", data);

        const markersGroup = [];
        const tabla = document.getElementById('tabla-mejor-ruta');
        const coordinatesForRoute = []; // Coordenadas para la API

        // Llenar marcadores y tabla
        data.ruta.forEach((punto, i) => {
            const latLng = [punto.lat, punto.lng];
            
            // Añadir marcador al mapa
            const marker = L.marker(latLng)
                .addTo(map)
                .bindPopup(`<b>${i + 1}. ${punto.nombre}</b><br>${punto.condicion}`);
            markersGroup.push(marker);

            // Añadir coordenadas para la ruta
            coordinatesForRoute.push(L.latLng(punto.lat, punto.lng));

            // Llenar la tabla
            const row = tabla.insertRow();
            row.insertCell(0).textContent = i + 1;
            row.insertCell(1).textContent = punto.nombre;
            row.insertCell(2).textContent = punto.condicion;
            row.cells[0].style.textAlign = "center";
        });

        // Ajustar el zoom del mapa
        if (markersGroup.length > 0) {
            map.fitBounds(L.featureGroup(markersGroup).getBounds().pad(0.2));
        }
        
        // Dibujar la ruta en el mapa
        dibujarRuta(data.ruta);

        // Mostrar información adicional
        console.log(`Fitness: ${data.fitness.toFixed(2)}`);
        console.log(`Tiempo de ejecución: ${data.tiempo_ejecucion.toFixed(2)}s`);

    } catch (error) {
        console.error("Error al procesar datos de ruta:", error);
        alert("Error al procesar los datos de la ruta");
    }

    // 5. Función para dibujar la ruta
    function dibujarRuta(coordenadas) {
        const urlCoordinates = coordenadas.map(c => `${c.lng},${c.lat}`).join(';');
        const apiUrl = `https://router.project-osrm.org/route/v1/driving/${urlCoordinates}?overview=full&geometries=geojson`;
        
        console.log("Llamando a la API de OSRM...");
        fetch(apiUrl)
            .then(res => res.json())
            .then(routeData => {
                if (routeData.code !== 'Ok') {
                    throw new Error("No se pudo obtener la ruta desde OSRM.");
                }
                
                // Usar GeoJSON directamente en lugar de polyline codificado
                const routeGeometry = routeData.routes[0].geometry;
                
                // Crear la polyline directamente desde las coordenadas GeoJSON
                const polyline = L.geoJSON(routeGeometry, {
                    style: {
                        color: 'black',
                        weight: 5,
                        opacity: 0.7
                    }
                }).addTo(map);

                console.log("¡Ruta dibujada correctamente!");
            })
            .catch(err => console.error("Error al dibujar la ruta:", err));
    }
    
    // 6. Funcionalidad de los botones
    document.getElementById("guardarRuta").addEventListener("click", () => {
        alert("Función para guardar ruta aún no implementada.");
    });

    document.getElementById("nuevaRuta").addEventListener("click", () => {
        // Limpiar sessionStorage y regresar a selección
        sessionStorage.removeItem('rutaOptimizada');
        window.location.href = '/';
    });
});
