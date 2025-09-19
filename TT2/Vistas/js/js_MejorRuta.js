// Inicializar el mapa
const map = L.map('mapa', {
    minZoom: 8,
    maxZoom: 14
}).setView([19.35, -99.75], 8); // centro aproximado del EdoMex

// Definir los límites (latitudes y longitudes) de la región permitida
const southWest = L.latLng(18.5, -100.5); // esquina inferior izquierda
const northEast = L.latLng(20.2, -98.5);  // esquina superior derecha
const bounds = L.latLngBounds(southWest, northEast);

// Aplicar los límites al mapa
map.setMaxBounds(bounds);
map.on('drag', function() {
    map.panInsideBounds(bounds, { animate: true });
});

// Capa base
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; CARTO',
    subdomains: 'abcd',
}).addTo(map);

// Cargar la ruta desde JSON generado por Python
fetch('ruta_Ejemplo.json')
    .then(res => res.json())
    .then(data => {
        // Crear grupo de marcadores para ajustar la vista
        const markersGroup = [];

        data.forEach((coord, i) => {
            const marker = L.marker(coord)
                .addTo(map)
                .bindPopup(`Punto ${i + 1}`);
            markersGroup.push(marker);
        });

        // Ajustar la vista inicial al grupo de marcadores, pero dentro de los límites
        const groupBounds = L.featureGroup(markersGroup).getBounds();
        if (bounds.contains(groupBounds)) {
            map.fitBounds(groupBounds);
        } else {
            map.fitBounds(bounds);
        }

        // Llenar la tabla con las rutas
        const tabla = document.getElementById('tabla-mejor-ruta');
        data.forEach((item, i) => {
            const row = tabla.insertRow();

            // Número de destino
            const cell0 = row.insertCell(0);
            cell0.textContent = i + 1;
            cell0.style.textAlign = "center";

            // Nombre de la nave
            const cell1 = row.insertCell(1);
            cell1.textContent = item.nombre;

            // Condición climática
            const cell2 = row.insertCell(2);
            cell2.textContent = item.condicion;
        });
    })
    .catch(err => console.error("No se pudo cargar ruta_Ejemplo.json:", err));

// Botón guardar ruta
document.getElementById("guardarRuta").addEventListener("click", () => {
    alert("Función para guardar ruta aún no implementada.");
});
