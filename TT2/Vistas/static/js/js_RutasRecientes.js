let lottieAnim = null;

document.addEventListener("DOMContentLoaded", async function() {
    // Inicializar animación Lottie
    const lottieContainer = document.getElementById('lottie-container');
    const overlay = document.getElementById('overlay-espera');

    if (typeof lottie !== 'undefined' && lottieContainer) {
        lottieAnim = lottie.loadAnimation({
            container: lottieContainer,
            renderer: 'svg',
            loop: true,
            autoplay: false,
            path: '/static/animaciones/PantallaEspera.json'
        });
    }

    // Botón cerrar sesión
    const cerrarSesionBtn = document.getElementById('cerrarSesion');
    if (cerrarSesionBtn) {
        cerrarSesionBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/cerrar-sesion', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }
                });
                if (response.ok) window.location.href = '/';
                else alert('Error al cerrar sesión');
            } catch (error) {
                console.error('Error:', error);
                alert('Error al cerrar sesión');
            }
        });
    }

    // Cargar rutas del usuario
    await cargarRutasUsuario();

    // ===============================
    // RE-HACER OPTIMIZACIÓN
    // ===============================
    const btnRehacer = document.getElementById("rehacerOptim");
    const loadingDiv = document.getElementById('loading-rutas');

    btnRehacer.addEventListener("click", async () => {
        const selected = document.querySelector('input[name="seleccionarRuta"]:checked');
        if (!selected) {
            alert("Selecciona una ruta primero.");
            return;
        }

        const rutaId = selected.value;

        try {
            // Mostrar overlay + animación
            if (overlay) overlay.classList.add("mostrar");
            if (lottieAnim) lottieAnim.play();

            btnRehacer.disabled = true;

            const response = await fetch(`/regenerar-ruta/${rutaId}`);
            const resultado = await response.json();

            if (response.ok) {
                sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
                window.location.href = '/mejor-ruta';
            } else {
                alert('Error: ' + resultado.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error al rehacer optimización.');
        } finally {
            // Ocultar overlay
            if (overlay) overlay.classList.remove("mostrar");
            if (lottieAnim) lottieAnim.stop();

            btnRehacer.disabled = false;
        }
    });

    // ===============================
    // Modal de detalles
    // ===============================
    const modal = document.getElementById('modal-detalles');
    const textoRutaModal = document.getElementById('modal-texto-ruta');
    const btnCerrar = document.getElementById('btn-cerrar-modal');

    const tbody = document.getElementById('tabla-rutas-body');
    
    tbody.addEventListener('click', async (event) => {

        // Ver más detalles
        if (event.target.classList.contains('btn-mas-detalles')) {
            const boton = event.target;
            const rutaCompleta = boton.dataset.rutaCompleta;

            textoRutaModal.textContent = rutaCompleta;
            modal.style.display = 'block';
        }

        // Eliminar ruta
        if (event.target.classList.contains('btn-eliminar') || event.target.parentElement.classList.contains('btn-eliminar')) {
            const boton = event.target.classList.contains('btn-eliminar') ? event.target : event.target.parentElement;
            const idRuta = boton.dataset.id;

            const confirmar = confirm("¿Estás seguro de que deseas eliminar esta ruta permanentemente?");
            if (!confirmar) return;

            try {
                const response = await fetch(`/eliminar-ruta/${idRuta}`, {
                    method: 'DELETE'
                });
                const data = await response.json();

                if (response.ok) {
                    alert("Ruta eliminada correctamente.");
                    await cargarRutasUsuario();
                } else {
                    alert("Error: " + data.message);
                }
            } catch (error) {
                console.error("Error al eliminar:", error);
                alert("Hubo un error al intentar eliminar la ruta.");
            }
        }
    });

    if(btnCerrar) {
        btnCerrar.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }

    window.addEventListener('click', (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    });
});

/* ########################################################################################################################### */ 
async function cargarRutasUsuario() {
    const loadingDiv = document.getElementById('loading-rutas');
    const noRutasDiv = document.getElementById('no-rutas');
    const tablaRutas = document.getElementById('tabla-rutas');
    const tbody = document.getElementById('tabla-rutas-body');
    const btnRehacer = document.getElementById("rehacerOptim");

    loadingDiv.style.display = 'block';
    noRutasDiv.style.display = 'none';
    tablaRutas.style.display = 'none';
    btnRehacer.disabled = true;

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

                    row.innerHTML = `
                        <td class="columna-indice">${index + 1}</td>
                        <td>${ruta.ruta_corta}</td>
                        <td style="text-align:center;">
                            <button class="btn-mas-detalles" data-ruta-completa="${ruta.ruta_completa}">
                                Ver más
                            </button>
                        </td>
                        <td style="text-align:center;">
                            <button class="btn-eliminar" data-id="${ruta.id_ruta}" title="Eliminar ruta">
                                Eliminar
                            </button>
                        </td>
                        <td style="text-align:center;">
                            <input type="radio" name="seleccionarRuta" value="${ruta.id_ruta}">
                        </td>
                    `;
                    tbody.appendChild(row);
                });

                const radios = document.querySelectorAll('input[name="seleccionarRuta"]');
                radios.forEach(r => {
                    r.addEventListener('change', () => {
                        btnRehacer.disabled = false;
                    });
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
