const tbody = document.getElementById("tabla-lugares");

// Leer nombres desde Nombres_Naves.csv
fetch("./Naves_Industriles.csv")
    .then(response => response.text())
    .then(data => {
        const lineas = data.split("\n").map(l => l.trim()).filter(l => l.length > 0);

        // Obtener índice de la columna "nombre" desde la cabecera
        const cabecera = lineas[0].split(",");
        const indiceNombre = cabecera.indexOf("nombre");

        // Generar filas usando solo la columna "nombre"
        lineas.slice(1).forEach((linea, i) => { // slice(1) para saltar la cabecera
            const columnas = linea.split(",");
            const nombre = columnas[indiceNombre].trim();

            let row = document.createElement("tr");
            let cell = document.createElement("td");

            let checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.id = "nave" + (i + 1);

            let label = document.createElement("label");
            label.htmlFor = "nave" + (i + 1);
            label.textContent = nombre;

            cell.appendChild(checkbox);
            cell.appendChild(label);
            row.appendChild(cell);
            tbody.appendChild(row);
        });
    })

// Botón generar ruta
document.getElementById("generarRuta").addEventListener("click", () => {
    let seleccionados = [];
    const checkboxes = document.querySelectorAll("input[type=checkbox]");

    checkboxes.forEach((check) => {
        if (check.checked) {
            let label = document.querySelector(`label[for=${check.id}]`);
            seleccionados.push(label.textContent);
        }
    });

    if (seleccionados.length >= 5) {
        alert("Generando ruta para: \n" + seleccionados.join(", "));
    } else {
        alert("Por favor selecciona al menos 5 naves industriales.");
    }
});
