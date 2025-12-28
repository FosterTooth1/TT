document.getElementById("registroForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    // Obtener datos del formulario
    let email = document.getElementById("email").value.trim();
    let nombre = document.getElementById("nombre").value.trim();
    let password = document.getElementById("password").value;
    let errorContainer = document.getElementById("password-errors");
    errorContainer.innerHTML = "";
    let errors = [];

    // Validar campos completos
    if (!email || !nombre || !password) {
        alert("Por favor, completa todos los campos.");
        return;
    }

    // Validar contraseña
    if (password.length < 8) errors.push("Debe tener al menos 8 caracteres.");
    if (!/[A-Z]/.test(password)) errors.push("Debe incluir al menos una letra mayúscula.");
    if (!/[a-z]/.test(password)) errors.push("Debe incluir al menos una letra minúscula.");
    if (!/\d/.test(password)) errors.push("Debe incluir al menos un número.");
    if (!/[!@#$%^&*()_\-+=\[\]{};':"\\|,.<>\/?]/.test(password))
        errors.push("Debe incluir al menos un carácter especial.");
    if (errors.length > 0) {
        errorContainer.innerHTML = errors.map(e => `<div>${e}</div>`).join("");
        return;
    }

    // Deshabilitar el botón de envío para evitar múltiples envíos
    const submitButton = this.querySelector('button[type="submit"]');
    submitButton.disabled = true;
    submitButton.innerHTML = "<b>Registrando...</b>";

    // Enviar datos al servidor
    try {
        const response = await fetch("/registrar_usuario", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                email: email,
                nombre: nombre,
                password: password
            })
        });

        const result = await response.json();
        // Manejar respuesta del servidor
        if (response.ok && result.status === "ok") {
            alert(result.message);
            window.location.href = "/rutas-recientes";
        } else {
            alert(result.message || "Error al crear la cuenta.");
        }
    } catch (err) {
        console.error("Error en el registro:", err);
        alert("Error de conexión con el servidor.");
    }finally {
        submitButton.disabled = false;
        submitButton.innerHTML = "<b>Registrarse</b>";
    }
});
