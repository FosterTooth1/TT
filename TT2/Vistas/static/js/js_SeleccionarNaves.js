const tbody = document.getElementById("tabla-lugares");
let navesData = [];

// Cargar naves desde la API
async function cargarNaves() {
    try {
        const response = await fetch('/api/naves');
        const naves = await response.json();
        
        if (response.ok) {
            navesData = naves; // Guardar datos para uso posterior
            // Generar filas usando los datos de la API
            naves.forEach((nave, i) => {
                let row = document.createElement("tr");
                let cell = document.createElement("td");

                let checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = "nave" + i;
                checkbox.value = i; // Usar el índice como valor

                // Cuando cambie una selección, actualizar el dropdown de inicio
                checkbox.addEventListener('change', actualizarDropdownInicio);

                let label = document.createElement("label");
                label.htmlFor = "nave" + i;
                label.textContent = nave.nombre;

                cell.appendChild(checkbox);
                cell.appendChild(label);
                row.appendChild(cell);
                tbody.appendChild(row);
            });
        } else {
            console.error("Error al cargar naves:", naves.error);
            alert("Error al cargar las naves industriales");
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error al conectar con el servidor");
    }
}

function actualizarDropdownInicio() {
    const startNaveContainer = document.getElementById('start-nave-container');
    const selectStartNave = document.getElementById('select-start-nave');
    
    // Si no existe el select/contendor, no hacemos nada
    if (!selectStartNave || !startNaveContainer) return;

    // Obtener naves seleccionadas
    const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]:checked");
    
    // Limpiar opciones anteriores
    const valorAnterior = selectStartNave.value; 
    selectStartNave.innerHTML = ''; 
    
    if (checkboxes.length > 0) {
        checkboxes.forEach(check => {
            const indice = parseInt(check.value);
            // Usar navesData para obtener el nombre (si existe)
            const nave = navesData[indice] || { nombre: `Nave ${indice}` };
            
            let option = document.createElement('option');
            option.value = indice; // El valor de la opción es el índice GLOBAL
            option.textContent = nave.nombre;
            selectStartNave.appendChild(option);
        });

        // Intentar re-seleccionar el valor anterior si aún está en la lista
        if (Array.from(selectStartNave.options).some(opt => opt.value === valorAnterior)) {
            selectStartNave.value = valorAnterior;
        }

        startNaveContainer.style.display = 'block'; // Mostrar contenedor
    } else {
        startNaveContainer.style.display = 'none'; // Ocultar si no hay naves
    }
}

// Manejo de cerrar sesión (reutilizable)
async function setupCerrarSesion() {
    const cerrarSesionBtn = document.getElementById('cerrarSesion');
    if (!cerrarSesionBtn) return;

    cerrarSesionBtn.addEventListener('click', async () => {
        try {
            cerrarSesionBtn.disabled = true;
            cerrarSesionBtn.textContent = 'Cerrando sesión...';

            const response = await fetch('/cerrar-sesion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (response.ok) {
                window.location.reload();
            } else {
                alert('Error al cerrar sesión');
                cerrarSesionBtn.disabled = false;
                cerrarSesionBtn.textContent = 'Cerrar Sesión';
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error al cerrar sesión');
            cerrarSesionBtn.disabled = false;
            cerrarSesionBtn.textContent = 'Cerrar Sesión';
        }
    });
}

// Seleccionar todo / deseleccionar todo
function setupSeleccionarTodo() {
    const boton = document.getElementById("seleccionarTodo");
    if (!boton) return;

    boton.addEventListener("click", () => {
        const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]");
        const allSelected = Array.from(checkboxes).every(checkbox => checkbox.checked);
        checkboxes.forEach((checkbox) => {
            checkbox.checked = !allSelected;
        });
        actualizarDropdownInicio();
    });
}

let lottieAnim = null;

// Inicialización principal al cargar la página
document.addEventListener("DOMContentLoaded", function() {
    cargarNaves();
    setupCerrarSesion();
    setupSeleccionarTodo();

    // Inicializar animación Lottie si el contenedor existe
    const lottieContainer = document.getElementById('lottie-container');
    if (typeof lottie !== 'undefined' && lottieContainer) {
        lottieAnim = lottie.loadAnimation({
            container: lottieContainer,
            renderer: 'svg',
            loop: true,
            autoplay: false,
            path: '/static/animaciones/PantallaEspera.json'
        });
    } else {
        // Si lottie no está cargado todavía, no fallamos; solo registramos null
        lottieAnim = null;
    }

    // Si hay un select-start-nave en el DOM y ya hay checkboxes pre-creados, actualizar dropdown
    // (esto es útil si recargas datos desde otro punto)
    actualizarDropdownInicio();
});

// ===============================
// BOTÓN "GENERAR RUTA" (con overlay + Lottie)
// ===============================
document.getElementById("generarRuta").addEventListener("click", async () => {
    let seleccionados = [];
    const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]:checked");

    checkboxes.forEach((check) => {
        seleccionados.push(parseInt(check.value));
    });

    if (seleccionados.length < 5) {
        alert("Por favor selecciona al menos 5 naves industriales.");
        return;
    }

    const selectStartNave = document.getElementById('select-start-nave');
    const indice_inicio = selectStartNave ? parseInt(selectStartNave.value) : null;

    // Validar que el índice de inicio sea un número válido si existe el select
    if (selectStartNave && (isNaN(indice_inicio) || indice_inicio === null)) {
        alert("Por favor selecciona una nave de inicio válida.");
        return;
    }

    // Preparar overlay y Lottie
    const overlay = document.getElementById("overlay-espera");
    try {
        if (overlay) {
            overlay.classList.add("mostrar");
        }
        if (lottieAnim && typeof lottieAnim.play === 'function') {
            lottieAnim.play();
        }

        // Deshabilitar botón para evitar dobles envíos
        const btn = document.getElementById("generarRuta");
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<b>Optimizando...</b>';
        }

        // Petición al backend (incluimos indice_inicio por si en el backend lo quieres usar luego)
        const payload = {
            indices: seleccionados
        };
        if (indice_inicio !== null) payload.indice_inicio = indice_inicio;

        const response = await fetch('/api/generar-ruta', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const resultado = await response.json();

        if (response.ok) {
            // Guardar resultado en sessionStorage para la siguiente página
            sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
            // Redirigir a la página de mejor ruta
            window.location.href = '/mejor-ruta';
            // Nota: la navegación hará unload de la página; el finally se ejecutará antes del unload
        } else {
            alert("Error: " + (resultado.error || 'Error al generar la ruta'));
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error al generar la ruta");
    } finally {
        // Restaurar estado de la UI (si la página ya fue redirigida esto no tiene efecto)
        if (overlay) {
            overlay.classList.remove("mostrar");
        }
        if (lottieAnim && typeof lottieAnim.stop === 'function') {
            lottieAnim.stop();
            // opcional: destruir si quieres liberar memoria
            // lottieAnim.destroy();
            // lottieAnim = null;
        }
        const btn = document.getElementById("generarRuta");
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<b>Optimizar Ruta</b>';
        }
    }
});
