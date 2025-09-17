const tbody = document.getElementById("tabla-lugares");

// Leer nombres desde naves.csv
fetch("./Nombres_Naves.csv")
    .then(response => response.text())
    .then(data => {
        const lineas = data.split("\n").map(l => l.trim()).filter(l => l.length > 0);

        // Generar filas con los nombres reales
        lineas.forEach((nombre, i) => {
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
    .catch(err => console.error("Error cargando naves.csv:", err));


// Botón generar ruta
document.getElementById("generarRuta").addEventListener("click", () => {
    let seleccionados = [];
    const checkboxes = document.querySelectorAll("input[type=checkbox]");

    checkboxes.forEach((check, i) => {
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
