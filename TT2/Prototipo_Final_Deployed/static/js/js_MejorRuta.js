document.addEventListener("DOMContentLoaded", function() {
    
    // Inicialización de Mapas
    const mapPenalizado = L.map('mapa-penalizado', { minZoom: 5 }).setView([19.35, -99.75], 8);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
    }).addTo(mapPenalizado);

    const mapLimpio = L.map('mapa-limpio', { minZoom: 5 }).setView([19.35, -99.75], 8);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
    }).addTo(mapLimpio);

    // Límites de mapa para evitar desplazamiento excesivo
    const bounds = L.latLngBounds(L.latLng(14.0, -118.0), L.latLng(32.0, -86.0));
    mapPenalizado.setMaxBounds(bounds);
    mapLimpio.setMaxBounds(bounds);

    // Variables para guardar los grupos de marcadores y usarlos al cambiar de pestaña
    let groupPenalizado = null;
    let groupLimpio = null;

    // Lógica de Pestañas
    const tabButtons = document.querySelectorAll('.tab-button');
    const mapContents = document.querySelectorAll('.mapa-wrapper');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Desactivar todos
            tabButtons.forEach(b => b.classList.remove('active'));
            mapContents.forEach(c => c.classList.remove('active'));

            // Activar actual
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            const targetContent = document.getElementById(targetId);
            targetContent.classList.add('active');

            // Recalcular tamaño y zoom al mostrar
            if (targetId === 'mapa-penalizado') {
                mapPenalizado.invalidateSize();
                if (groupPenalizado) {
                    mapPenalizado.fitBounds(groupPenalizado.getBounds().pad(0.1));
                }
            } else if (targetId === 'mapa-limpio') {
                mapLimpio.invalidateSize();
                if (groupLimpio) {
                    mapLimpio.fitBounds(groupLimpio.getBounds().pad(0.1));
                }
            }
        });
    });

    // Iconos personalizados
    const iconoVerde = new L.Icon({
        iconUrl: '/static/images/marker-icon-2x-green.png',
        shadowUrl: '/static/images/marker-shadow.png',
        iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
    });
    const iconoAzul = new L.Icon.Default(); 
    const iconoRojo = new L.Icon({
        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
        iconSize: [25, 41], iconAnchor: [12, 41], popupAnchor: [1, -34], shadowSize: [41, 41]
    });

    // Procesar datos de ruta desde sessionStorage
    const rutaDataStr = sessionStorage.getItem('rutaOptimizada');
    if (!rutaDataStr) {
        alert("No hay datos de ruta.");
        window.location.href = '/';
        return;
    }

    try {
        const data = JSON.parse(rutaDataStr);

        // Actualizar Estadísticas
        if (document.getElementById('fitness-valor')) 
            document.getElementById('fitness-valor').textContent = data.fitness_penalizado ? data.fitness_penalizado.toFixed(2) : "N/A";
        
        if (document.getElementById('distancia-real-valor')) 
            document.getElementById('distancia-real-valor').textContent = data.distancia_real ? data.distancia_real.toFixed(2) : "N/A";

        const climaModaEl = document.getElementById('clima-moda');
        if (climaModaEl && data.ruta_penalizada) {
            const condiciones = data.ruta_penalizada.map(p => p.condicion).filter(c => c);
            if (condiciones.length > 0) {
                const conteo = condiciones.reduce((acc, val) => { acc[val] = (acc[val] || 0) + 1; return acc; }, {});
                const moda = Object.keys(conteo).reduce((a, b) => conteo[a] > conteo[b] ? a : b);
                climaModaEl.textContent = moda;
            } else { climaModaEl.textContent = "N/A"; }
        }

        // Renderizar Mapa 1 (Penalizado)
        const markersPenalizados = [];
        const tabla = document.getElementById('tabla-mejor-ruta');
        tabla.innerHTML = ""; 

        data.ruta_penalizada.forEach((punto, i) => {
            const latLng = [punto.lat, punto.lng];
            const marker = L.marker(latLng, { icon: (i === 0) ? iconoVerde : iconoAzul })
                .addTo(mapPenalizado)
                .bindPopup(`<b>${i + 1}. ${punto.nombre}</b><br>${punto.condicion}`);
            markersPenalizados.push(marker);

            const row = tabla.insertRow();
            row.insertCell(0).textContent = i + 1;
            row.insertCell(1).textContent = punto.nombre;
            row.insertCell(2).textContent = punto.condicion;
        });

        if (markersPenalizados.length > 0) {
            groupPenalizado = L.featureGroup(markersPenalizados);
            mapPenalizado.fitBounds(groupPenalizado.getBounds().pad(0.1));
            dibujarRuta(data.ruta_penalizada, mapPenalizado, '#08564d');
        }

        // Renderizar Mapa 2 (Sin penalizaciones)
        const markersLimpios = [];
        if (data.ruta_limpia) {
            data.ruta_limpia.forEach((punto, i) => {
                const latLng = [punto.lat, punto.lng];
                let iconToUse = iconoAzul;
                let msgComparativa = "";
                
                if (i === 0) {
                    iconToUse = iconoVerde;
                } else {
                    const nombreEnPenalizada = data.ruta_penalizada[i] ? data.ruta_penalizada[i].nombre : null;
                    if (nombreEnPenalizada !== punto.nombre) {
                        iconToUse = iconoRojo;
                        msgComparativa = "<br><span style='color:red'>⚠️ Posición diferente</span>";
                    }
                }

                const marker = L.marker(latLng, { icon: iconToUse })
                    .addTo(mapLimpio)
                    .bindPopup(`<b>${i + 1}. ${punto.nombre}</b>${msgComparativa}`);
                markersLimpios.push(marker);
            });

            if (markersLimpios.length > 0) {
                groupLimpio = L.featureGroup(markersLimpios);
                dibujarRuta(data.ruta_limpia, mapLimpio, '#417fc6');
            }
        }
        
        // Renderizar Gráfica de Fitness limitado a dos decimales
        renderizarGrafica(data.fitness_generaciones.map(f => f.toFixed(2)));

    } catch (error) {
        console.error("Error JS:", error);
    }

    // Función para dibujar la ruta usando OSRM
    function dibujarRuta(coordenadas, mapaObj, colorLinea) {
        const coordsParaRuta = [...coordenadas];
        if (coordsParaRuta.length > 0) coordsParaRuta.push(coordsParaRuta[0]);
        const urlCoordinates = coordsParaRuta.map(c => `${c.lng},${c.lat}`).join(';');
        fetch(`https://router.project-osrm.org/route/v1/driving/${urlCoordinates}?overview=full&geometries=geojson`)
            .then(res => res.json())
            .then(routeData => {
                if (routeData.code === 'Ok') {
                    L.geoJSON(routeData.routes[0].geometry, {
                        style: { color: colorLinea, weight: 4, opacity: 0.8 }
                    }).addTo(mapaObj);
                }
            }).catch(e => console.error(e));
    }

    // Función para renderizar la gráfica de fitness
    function renderizarGrafica(fitnessData) {
        const graficaEl = document.getElementById('grafica-fitness');
        if (graficaEl && fitnessData) {
            new Chart(graficaEl.getContext('2d'), {
                type: 'line',
                data: {
                    labels: fitnessData.map((_, i) => i + 1),
                    datasets: [{
                        label: 'Costo', data: fitnessData,
                        borderColor: '#08564d', backgroundColor: 'rgba(8, 86, 77, 0.1)',
                        fill: true, tension: 0.1
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        }
    }

    // Eventos de Botones
    document.getElementById("guardarRuta")?.addEventListener("click", async () => {
        const d = JSON.parse(sessionStorage.getItem('rutaOptimizada'));
        const body = { destinos: d.ruta_penalizada, indices: d.indices_penalizada };
        try {
            const res = await fetch('/guardar-ruta', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
            });
            alert(res.ok ? "Ruta guardada" : "Error al guardar");
        } catch (e) { alert("Error de conexión"); }
    });
    document.getElementById("nuevaRuta")?.addEventListener("click", () => window.location.href = '/');
    document.getElementById('cerrarSesion')?.addEventListener('click', async () => {
         await fetch('/cerrar-sesion', { method: 'POST' }); window.location.reload();
    });
});