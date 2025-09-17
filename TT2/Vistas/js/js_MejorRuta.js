// Inicializar el mapa
const map = L.map('mapa').setView([20, -100], 6); // centrado aproximado en México

// Capa base
L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; CARTO',
    subdomains: 'abcd',
    minZoom: 5,
    maxZoom: 7
}).addTo(map);

// Cargar la ruta desde JSON generado por Python
fetch('ruta_Ejemplo.json')
    .then(res => res.json())
    .then(data => {
        // Agregar marcadores para cada punto
        data.forEach((coord, i) => {
            L.marker(coord).addTo(map)
                .bindPopup(`Punto ${i+1}`);
        });

        // Ajustar vista al bounds de los marcadores
        const group = L.featureGroup(data.map(coord => L.marker(coord)));
        map.fitBounds(group.getBounds());

        // Llenar la tabla con las rutas
        const tabla = document.getElementById('tabla-mejor-ruta');
        data.forEach((coord, i) => {
            // Primer celda: Numero de destino
            const row = tabla.insertRow();
            const cell0 = row.insertCell(0);
            cell0.textContent = i + 1;
            cell0.style.textAlign = "center";

            // Segunda celda: coordenadas
            const cell1 = row.insertCell(1);
            cell1.textContent = `Lat: ${coord[0]}, Lng: ${coord[1]}`;
        });


    })
    .catch(err => console.error("No se pudo cargar ruta.json:", err));

// Botón guardar ruta
document.getElementById("guardarRuta").addEventListener("click", () => {
    alert("Función para guardar ruta aún no implementada.");
});







