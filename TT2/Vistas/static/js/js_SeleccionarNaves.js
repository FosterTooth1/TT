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
    
    // Obtener naves seleccionadas
    const checkboxes = document.querySelectorAll("input[type=checkbox]:checked");
    
    // Limpiar opciones anteriores
    selectStartNave.innerHTML = ''; 
    
    if (checkboxes.length > 0) {
        // Guardar el valor seleccionado anteriormente, si existe
        const valorAnterior = selectStartNave.value; 
        
        checkboxes.forEach(check => {
            const indice = parseInt(check.value);
            // Usar navesData para obtener el nombre
            const nave = navesData[indice]; 
            
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

// Cargar naves al cargar la página
document.addEventListener("DOMContentLoaded", function() {
    cargarNaves();
    
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
                    // Recargar la página para actualizar la interfaz
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

// Botón seleccionar todo
document.getElementById("seleccionarTodo").addEventListener("click", () => {
    const checkboxes = document.querySelectorAll("#tabla-lugares input[type=checkbox]");
    const allSelected = Array.from(checkboxes).every(checkbox => checkbox.checked);
    checkboxes.forEach((checkbox) => {
        checkbox.checked = !allSelected;
    });
    actualizarDropdownInicio();
});

// Botón generar ruta
// Botón generar ruta
document.getElementById("generarRuta").addEventListener("click", async () => {
    let seleccionados = [];
    const checkboxes = document.querySelectorAll("input[type=checkbox]:checked");

    checkboxes.forEach((check) => {
        seleccionados.push(parseInt(check.value));
    });

    if (seleccionados.length < 5) {
        alert("Por favor selecciona al menos 5 naves industriales.");
        return;
    }

    const selectStartNave = document.getElementById('select-start-nave');
    const indice_inicio = parseInt(selectStartNave.value);

    // Validar que el índice de inicio sea un número válido
    if (isNaN(indice_inicio)) {
        alert("Por favor selecciona una nave de inicio válida.");
        return;
    }

    // Mostrar loading
    document.getElementById("loading").style.display = "block";
    document.getElementById("generarRuta").disabled = true;

    try {
        const response = await fetch('/api/generar-ruta', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                indices: seleccionados,
                indice_inicio: indice_inicio 
            })
        });

        const resultado = await response.json();

        if (response.ok) {
            // Guardar resultado en sessionStorage para la siguiente página
            sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
            // Redirigir a la página de mejor ruta
            window.location.href = '/mejor-ruta';
        } else {
            alert("Error: " + resultado.error);
        }
    } catch (error) {
        console.error("Error:", error);
        alert("Error al generar la ruta");
    } finally {
        // Ocultar loading
        document.getElementById("loading").style.display = "none";
        document.getElementById("generarRuta").disabled = false;
    }
});