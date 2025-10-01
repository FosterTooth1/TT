document.getElementById("registroForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value;
    let nombre = document.getElementById("nombre").value;
    let password = document.getElementById("password").value;

    if (email && nombre && password) {
        alert("¡Cuenta creada exitosamente! Bienvenido, " + nombre);
        // Aquí se implementaría la lógica de registro real
        // Por ahora, redirigir a rutas recientes
        window.location.href = '/rutas-recientes';
    } else {
        alert("Por favor, completa todos los campos.");
    }
});
