document.getElementById("registroForm").addEventListener("submit", function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value;
    let nombre = document.getElementById("nombre").value;
    let password = document.getElementById("password").value;
    let errorContainer = document.getElementById("password-errors");
    errorContainer.innerHTML = ""; // Limpiar errores anteriores

    // Validaciones individuales
    let errors = [];

    if (!email || !nombre || !password) {
        alert("Por favor, completa todos los campos.");
        return;
    }

    if (password.length < 8) {
        errors.push("Debe tener al menos 8 caracteres.");
    }
    if (!/[A-Z]/.test(password)) {
        errors.push("Debe incluir al menos una letra mayúscula.");
    }
    if (!/[a-z]/.test(password)) {
        errors.push("Debe incluir al menos una letra minúscula.");
    }
    if (!/\d/.test(password)) {
        errors.push("Debe incluir al menos un número.");
    }
    if (!/[!@#$%^&*()_\-+=\[\]{};':"\\|,.<>\/?]/.test(password)) {
        errors.push("Debe incluir al menos un carácter especial.");
    }

    if (errors.length > 0) {
        // Mostrar errores en la vista
        errorContainer.innerHTML = errors.map(e => `<div>${e}</div>`).join("");
        return;
    }

    // Si pasa todas las validaciones
    alert("¡Cuenta creada exitosamente! Bienvenido, " + nombre);
    window.location.href = '/rutas-recientes';
});
