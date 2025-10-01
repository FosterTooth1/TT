const tbody = document.getElementById("tabla-lugares");

// Cargar naves desde la API
async function cargarNaves() {
    try {
        const response = await fetch('/api/naves');
        const naves = await response.json();
        
        if (response.ok) {
            // Generar filas usando los datos de la API
            naves.forEach((nave, i) => {
                let row = document.createElement("tr");
                let cell = document.createElement("td");

                let checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.id = "nave" + i;
                checkbox.value = i; // Usar el índice como valor

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

// Cargar naves al cargar la página
document.addEventListener("DOMContentLoaded", cargarNaves);

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

    // Mostrar loading
    document.getElementById("loading").style.display = "block";
    document.getElementById("generarRuta").disabled = true;

    try {
        const response = await fetch('/api/generar-ruta', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ indices: seleccionados })
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
