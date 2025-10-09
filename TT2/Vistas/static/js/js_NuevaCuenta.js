document.getElementById("registroForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    let email = document.getElementById("email").value.trim();
    let nombre = document.getElementById("nombre").value.trim();
    let password = document.getElementById("password").value;
    let errorContainer = document.getElementById("password-errors");
    errorContainer.innerHTML = "";

    let errors = [];

    if (!email || !nombre || !password) {
        alert("Por favor, completa todos los campos.");
        return;
    }

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

    // Enviar los datos al backend Flask
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

        if (response.ok && result.status === "ok") {
            alert(result.message);
            // Redirigir a rutas recientes ya que la sesión se estableció automáticamente
            window.location.href = "/rutas-recientes";
        } else {
            alert(result.message || "Error al crear la cuenta.");
        }
    } catch (err) {
        console.error("Error en el registro:", err);
        alert("Error de conexión con el servidor.");
    }
});
