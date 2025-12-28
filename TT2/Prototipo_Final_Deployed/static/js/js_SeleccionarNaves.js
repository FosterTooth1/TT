const tbody = document.getElementById("tabla-lugares");
let navesData = [];
let lottieAnim = null;

// Cargar naves industriales desde el servidor
async function cargarNaves() {
    try {
        const response = await fetch('/api/naves');
        const naves = await response.json();
        
        if (response.ok) {
            navesData = naves;

            naves.forEach((nave, i) => {
                let row = document.createElement("tr");
                let cell = document.createElement("td");

                let checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = "nave" + i;
                checkbox.value = i;

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

// Actualizar el dropdown de selección de nave de inicio
function actualizarDropdownInicio() {

    const startNaveContainer = document.getElementById('start-nave-container');
    const selectStartNave = document.getElementById('select-start-nave');
    
    if (!selectStartNave || !startNaveContainer) return;

    const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]:checked");

    const valorAnterior = selectStartNave.value;
    selectStartNave.innerHTML = '';

    if (checkboxes.length > 0) {
        checkboxes.forEach(check => {
            const indice = parseInt(check.value);
            const nave = navesData[indice] || { nombre: `Nave ${indice}` };

            let option = document.createElement('option');
            option.value = indice;
            option.textContent = nave.nombre;
            selectStartNave.appendChild(option);
        });


        if (Array.from(selectStartNave.options).some(opt => opt.value === valorAnterior)) {
            selectStartNave.value = valorAnterior;
        }

        startNaveContainer.style.display = 'block';
    } else {
        startNaveContainer.style.display = 'none';
    }
}

// Cerrar sesión
function setupCerrarSesion() {
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
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error al cerrar sesión');
        }

        cerrarSesionBtn.disabled = false;
        cerrarSesionBtn.textContent = 'Cerrar Sesión';
    });
}

// Seleccionar/Deseleccionar todo
function setupSeleccionarTodo() {
    const boton = document.getElementById("seleccionarTodo");
    if (!boton) return;

    boton.addEventListener("click", () => {
        const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]");
        const allSelected = Array.from(checkboxes).every(checkbox => checkbox.checked);

        checkboxes.forEach(checkbox => checkbox.checked = !allSelected);
        actualizarDropdownInicio();
    });
}

// Inicialización al cargar el DOM
document.addEventListener("DOMContentLoaded", function () {

    cargarNaves();
    setupCerrarSesion();
    setupSeleccionarTodo();

    // Inicializar animación Lottie
    const lottieContainer = document.getElementById('lottie-container');
    if (typeof lottie !== 'undefined' && lottieContainer) {
        lottieAnim = lottie.loadAnimation({
            container: lottieContainer,
            renderer: 'svg',
            loop: true,
            autoplay: false,
            path: '/static/animations/PantallaEspera.json'
        });
    }

    actualizarDropdownInicio();

    // Evento para el botón Generar Ruta
    const btnGenerar = document.getElementById("generarRuta");

    if (btnGenerar) {
        btnGenerar.addEventListener("click", async () => {
            let seleccionados = [];
            const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]:checked");

            checkboxes.forEach(check => seleccionados.push(parseInt(check.value)));

            if (seleccionados.length < 5) {
                alert("Por favor selecciona al menos 5 naves industriales.");
                return;
            }

            const selectStartNave = document.getElementById('select-start-nave');
            const indice_inicio = selectStartNave ? parseInt(selectStartNave.value) : null;

            if (selectStartNave && (isNaN(indice_inicio) || indice_inicio === null)) {
                alert("Por favor selecciona una nave de inicio válida.");
                return;
            }

            const overlay = document.getElementById("overlay-espera");

            try {
                if (overlay) overlay.classList.add("mostrar");
                if (lottieAnim && typeof lottieAnim.play === 'function') lottieAnim.play();

                btnGenerar.disabled = true;
                btnGenerar.innerHTML = '<b>Optimizando...</b>';

                const payload = { indices: seleccionados };
                if (indice_inicio !== null) payload.indice_inicio = indice_inicio;

                const response = await fetch('/api/generar-ruta', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const resultado = await response.json();

                if (response.ok) {
                    sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
                    window.location.href = '/mejor-ruta';
                } else {
                    alert("Error: " + (resultado.error || 'Error al generar la ruta'));
                }
            } catch (error) {
                console.error("Error:", error);
                alert("Error al generar la ruta");
            } finally {
                if (overlay) overlay.classList.remove("mostrar");
                if (lottieAnim && typeof lottieAnim.stop === 'function') lottieAnim.stop();

                btnGenerar.disabled = false;
                btnGenerar.innerHTML = '<b>Optimizar Ruta</b>';
            }
        });
    }
});
