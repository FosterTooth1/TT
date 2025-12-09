document.addEventListener("DOMContentLoaded", function() {
    // 1. Inicializar el mapa
    const map = L.map('mapa', {
        minZoom: 8,
        maxZoom: 14
    }).setView([19.35, -99.75], 8);

    // 2. Definir y aplicar límites geográficos
    const northEast = L.latLng(20.35, -98.5);
    const southWest = L.latLng(18.5, -100.5);
    const bounds = L.latLngBounds(southWest, northEast);
    map.setMaxBounds(bounds);
    map.on('drag', () => map.panInsideBounds(bounds, { animate: true }));

    // 3. Añadir capa base del mapa (el fondo)
    L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> &copy; CARTO',
    }).addTo(map);

    /* ########################################################################################################################### */ 
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

        // // 1. Mostrar Fitness Final
        const fitnessEl = document.getElementById('fitness-valor');
        if (fitnessEl && data.fitness) {
            // .toFixed(2) para redondear a 2 decimales
            fitnessEl.textContent = data.fitness.toFixed(2);
        }

        // 2. Calcular y Mostrar Moda del Clima
        const climaModaEl = document.getElementById('clima-moda');
        if (climaModaEl && data.ruta) {
            // Obtenemos todas las condiciones, filtrando valores nulos o vacíos
            const condiciones = data.ruta.map(punto => punto.condicion).filter(c => c); 
            
            if (condiciones.length > 0) {
                // Contamos la frecuencia de cada condición
                const conteo = condiciones.reduce((acc, val) => {
                    acc[val] = (acc[val] || 0) + 1;
                    return acc;
                }, {});
                // Encontramos la clave (condición) con el valor (conteo) más alto
                const moda = Object.keys(conteo).reduce((a, b) => conteo[a] > conteo[b] ? a : b);
                climaModaEl.textContent = moda;
            } else {
                climaModaEl.textContent = "No disponible";
            }
        }

        // 3. Generar Gráfica de Evolución del Fitness
        const graficaEl = document.getElementById('grafica-fitness');
        // Verificamos que el elemento exista y que tengamos los datos
        if (graficaEl && data.fitness_generaciones && data.fitness_generaciones.length > 0) {
            const ctx = graficaEl.getContext('2d');
            // Creamos etiquetas para el eje X (1, 2, 3, ... N)
            const labels = Array.from({ length: data.fitness_generaciones.length }, (_, i) => i + 1);
            
            new Chart(ctx, {
                type: 'line', // Tipo de gráfica
                data: {
                    labels: labels, // Eje X (Generaciones)
                    datasets: [{
                        label: 'Fitness (Distancia)',
                        data: data.fitness_generaciones, // Eje Y (Valores de fitness)
                        borderColor: '#000000', // Color de la línea
                        backgroundColor: 'rgba(0, 0, 0, 0.1)', // Relleno bajo la línea
                        fill: true,
                        tension: 0.1 // Curvatura de la línea
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false, // Permitir que la gráfica no sea cuadrada
                    plugins: {
                        legend: {
                            display: false // Ocultar la leyenda "Fitness (Distancia)"
                        },
                        title: {
                            display: true,
                            text: 'Evolución del Fitness por Generación' // Título de la gráfica
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Generación' // Etiqueta Eje X
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Fitness (Distancia)' // Etiqueta Eje Y
                            }
                        }
                    }
                }
            });
        }

        console.log("Índices originales:", data.indices);
        const indicesRuta = data.indices; // Indices de la ruta seleccionada por el usuario

        const markersGroup = [];
        const tabla = document.getElementById('tabla-mejor-ruta');
        const coordinatesForRoute = []; // Coordenadas para la API

        const iconoInicio = new L.Icon({
            iconUrl: '/static/imagenes/marker-icon-2x-green.png',
            shadowUrl: '/static/imagenes/marker-shadow.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });

        // Llenar marcadores y tabla
        data.ruta.forEach((punto, i) => {
            const latLng = [punto.lat, punto.lng];
            
            let marker;

            // Si es el índice 0 (Inicio), usar el icono verde
            if (i === 0) {
                marker = L.marker(latLng, { icon: iconoInicio })
                    .addTo(map)
                    .bindPopup(`<b>🚩 INICIO: ${punto.nombre}</b><br>${punto.condicion}`);
            } else {
                // Para el resto, usar el marcador azul por defecto
                marker = L.marker(latLng)
                    .addTo(map)
                    .bindPopup(`<b>${i + 1}. ${punto.nombre}</b><br>${punto.condicion}`);
            }

            markersGroup.push(marker);

            // Añadir coordenadas para la ruta
            coordinatesForRoute.push(L.latLng(punto.lat, punto.lng));

            // Llenar la tabla
            const row = tabla.insertRow();
            // (El resto del código de la tabla sigue igual...)
             row.insertCell(0).textContent = i + 1;
             row.insertCell(1).textContent = punto.nombre;
             row.insertCell(2).textContent = punto.condicion;
             row.cells[0].style.textAlign = "center";
        });

        if (markersGroup.length > 0) {
            map.fitBounds(L.featureGroup(markersGroup).getBounds().pad(0.2)); // Ajustar el zoom del mapa
        }
        dibujarRuta(data.ruta); // Dibujar la ruta en el mapa

    } catch (error) {
        console.error("Error al procesar datos de ruta:", error);
        alert("Error al procesar los datos de la ruta");
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