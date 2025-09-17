document.getElementById("registroForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value;
    let nombre = document.getElementById("nombre").value;
    let password = document.getElementById("password").value;

    if (email && nombre && password) {
        alert("¡Registro exitoso! Bienvenido, " + nombre);
    } else {
        alert("Por favor, completa todos los campos.");
    }
});
