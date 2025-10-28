document.getElementById("loginForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const submitButton = document.getElementById("loginForm").querySelector('button[type="submit"]');

    let email = document.getElementById("email").value.trim();
    let password = document.getElementById("password").value;

    if (!email || !password) {
        alert("Por favor, completa todos los campos.");
        return;
    }

    if (submitButton) {
        submitButton.disabled = true;
    }

    try {
        const response = await fetch("/login_usuario", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: email, password: password })
        });

        const result = await response.json();

        if (response.ok && result.status === "ok") {
            alert(result.message);
            window.location.href = result.redirect_url || "/";
        } else {
            alert(result.message || "Error al iniciar sesión.");
        }
    } catch (err) {
        console.error("Error en la solicitud:", err);
        alert("Error de conexión con el servidor.");
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
        }
    }
});