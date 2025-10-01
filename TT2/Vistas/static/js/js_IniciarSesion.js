document.getElementById("registroForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    if (email && password) {
        alert("¡Inicio de sesión exitoso! Bienvenido, " + email);
        // Aquí se implementaría la lógica de autenticación real
        // Por ahora, redirigir a rutas recientes
        window.location.href = '/rutas-recientes';
    } else {
        alert("Por favor, completa todos los campos.");
    }
});
