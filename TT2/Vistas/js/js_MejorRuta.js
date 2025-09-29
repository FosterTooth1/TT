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

    // 4. Cargar los datos del archivo JSON
    fetch('ruta_Ejemplo.json')
        .then(response => {
            if (!response.ok) {
                throw new Error(`Error al cargar el archivo JSON: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("JSON cargado:", data);

            const markersGroup = [];
            const tabla = document.getElementById('tabla-mejor-ruta');
            const coordinatesForRoute = []; // Coordenadas para la API

            // Llenar marcadores y tabla
            data.forEach((punto, i) => {
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
            dibujarRuta(data);
        })
        .catch(err => {
            console.error("Error en la carga de datos:", err);
        });

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
    
    // 6. Funcionalidad del botón
    document.getElementById("guardarRuta").addEventListener("click", () => {
        alert("Función para guardar ruta aún no implementada.");
    });
});