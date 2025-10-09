document.addEventListener("DOMContentLoaded", async function() {
    // Botón para generar nueva ruta
    document.getElementById("generarNuevaRuta").addEventListener("click", () => {
        window.location.href = '/';
    });
    
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
                    // Redirigir a la página principal
                    window.location.href = '/';
                } else {
                    alert('Error al cerrar sesión');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error al cerrar sesión');
            }
        });
    }

    // Cargar rutas del usuario
    await cargarRutasUsuario();
});

/* ########################################################################################################################### */ 
async function cargarRutasUsuario() {
    const loadingDiv = document.getElementById('loading-rutas');
    const noRutasDiv = document.getElementById('no-rutas');
    const tablaRutas = document.getElementById('tabla-rutas');
    const tbody = document.getElementById('tabla-rutas-body');
    
    loadingDiv.style.display = 'block';
    noRutasDiv.style.display = 'none';
    tablaRutas.style.display = 'none';

    try {
        const response = await fetch('/obtener_rutas');
        const data = await response.json();

        loadingDiv.style.display = 'none';

        if (data.status === "ok") {
            if (data.rutas.length === 0) {
                noRutasDiv.style.display = 'block';
            } else {
                tablaRutas.style.display = 'table';
                tbody.innerHTML = "";

                data.rutas.forEach((ruta, index) => {
                    const row = document.createElement('tr');
                    const detallesCompletos = `${ruta.detalles}`;
                    row.innerHTML = `
                        <td>${detallesCompletos}</td>
                    `;
                    tbody.appendChild(row);
                });
            }
        } else {
            noRutasDiv.innerHTML = `<p>Error: ${data.message}</p>`;
            noRutasDiv.style.display = 'block';
        }
    } catch (err) {
        console.error("Error al cargar rutas:", err);
        loadingDiv.style.display = 'none';
        noRutasDiv.innerHTML = `<p>Error al conectar con el servidor.</p>`;
        noRutasDiv.style.display = 'block';
    }
}

/* ########################################################################################################################### */ 
async function verRuta(idRuta) {
    try {
        const response = await fetch(`/regenerar-ruta/${idRuta}`);
        const resultado = await response.json();
        
        if (response.ok) {
            // Guardar resultado en sessionStorage para la página de mejor ruta
            sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
            // Redirigir a la página de mejor ruta
            window.location.href = '/mejor-ruta';
        } else {
            alert('Error: ' + resultado.message);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error al cargar la ruta');
    }
}
