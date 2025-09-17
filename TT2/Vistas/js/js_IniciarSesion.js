document.getElementById("registroForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value;
    let password = document.getElementById("password").value;

    if (email && password) {
        alert("¡Inicio de sesión exitoso! Bienvenido, " + email);
    } else {
        alert("Por favor, completa todos los campos.");
    }
});
