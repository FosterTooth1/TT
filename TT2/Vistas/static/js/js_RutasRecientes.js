document.addEventListener("DOMContentLoaded", async function() {
   // Botón cerrar sesión
    const cerrarSesionBtn = document.getElementById('cerrarSesion');
    if (cerrarSesionBtn) {
        cerrarSesionBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/cerrar-sesion', { 
                    method: 'POST', 
                    headers: { 'Content-Type': 'application/json' }
                });
                if (response.ok) window.location.href = '/';
                else alert('Error al cerrar sesión');
            } catch (error) {
                console.error('Error:', error);
                alert('Error al cerrar sesión');
            }
        });
    }

    // Cargar rutas del usuario
    await cargarRutasUsuario();

    // Evento para "Re-hacer optimización"
    const btnRehacer = document.getElementById("rehacerOptim");
    const loadingDiv = document.getElementById('loading-rutas'); 

    btnRehacer.addEventListener("click", async () => {
        const selected = document.querySelector('input[name="seleccionarRuta"]:checked');
        if (!selected) {
            alert("Selecciona una ruta primero.");
            return;
        }

        const rutaId = selected.value;
        try {
            // Mostrar loading y deshabilitar botón
            loadingDiv.style.display = "block";
            btnRehacer.disabled = true;

            const response = await fetch(`/regenerar-ruta/${rutaId}`);
            const resultado = await response.json();

            if (response.ok) {
                // Guardar resultado en sessionStorage y redirigir
                sessionStorage.setItem('rutaOptimizada', JSON.stringify(resultado));
                window.location.href = '/mejor-ruta';
            } else {
                alert('Error: ' + resultado.message);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Error al rehacer optimización.');
        } finally {
            // Ocultar loading y reactivar botón solo si hay error
            loadingDiv.style.display = "none";
            btnRehacer.disabled = false;
        }
    });

    const modal = document.getElementById('modal-detalles');
    const textoRutaModal = document.getElementById('modal-texto-ruta');
    const btnCerrar = document.getElementById('btn-cerrar-modal');

    const tbody = document.getElementById('tabla-rutas-body');
    
    tbody.addEventListener('click', async (event) => {
    
    // Lógica para "Ver más"
    if (event.target.classList.contains('btn-mas-detalles')) {
        const boton = event.target;
        const rutaCompleta = boton.dataset.rutaCompleta;
        const textoRutaModal = document.getElementById('modal-texto-ruta');
        const modal = document.getElementById('modal-detalles');
        
        textoRutaModal.textContent = rutaCompleta;
        modal.style.display = 'block';
    }

    // Lógica para "Eliminar"
    if (event.target.classList.contains('btn-eliminar') || event.target.parentElement.classList.contains('btn-eliminar')) {
        // Manejar click tanto en el botón como en el icono
        const boton = event.target.classList.contains('btn-eliminar') ? event.target : event.target.parentElement;
        const idRuta = boton.dataset.id;

        // Mostrar confirmación
        const confirmar = confirm("¿Estás seguro de que deseas eliminar esta ruta permanentemente?");

        if (confirmar) {
            try {
                // Hacer petición al servidor
                const response = await fetch(`/eliminar-ruta/${idRuta}`, {
                    method: 'DELETE'
                });
                const data = await response.json();

                if (response.ok) {
                    alert("Ruta eliminada correctamente.");
                    // Recargar la tabla para reflejar cambios
                    await cargarRutasUsuario();
                } else {
                    alert("Error: " + data.message);
                }
            } catch (error) {
                console.error("Error al eliminar:", error);
                alert("Hubo un error al intentar eliminar la ruta.");
            }
        }
    }
});

    if(btnCerrar) {
        btnCerrar.addEventListener('click', () => {
            modal.style.display = 'none';
        });
    }

    window.addEventListener('click', (event) => {
        if (event.target == modal) {
            modal.style.display = 'none';
        }
    });
});

// =========================================================
    // NUEVO: LÓGICA DEL CHATBOT
    // =========================================================
    const openChatBtn = document.getElementById('open-chat-btn');
    const closeChatBtn = document.getElementById('close-chat-btn');
    const chatWindow = document.getElementById('chat-window');
    const chatInput = document.getElementById('chat-input');
    const sendChatBtn = document.getElementById('send-chat-btn');
    const messagesContainer = document.getElementById('chat-messages');

    // Abrir/Cerrar Chat
    openChatBtn.addEventListener('click', () => {
        chatWindow.classList.remove('chat-hidden');
        openChatBtn.style.display = 'none';
    });

    closeChatBtn.addEventListener('click', () => {
        chatWindow.classList.add('chat-hidden');
        openChatBtn.style.display = 'flex';
    });

    // Función para obtener lo que ve el usuario (Tabla)
    function obtenerContextoVisual() {
        const filas = document.querySelectorAll("#tabla-rutas-body tr");
        let ids_en_pantalla = [];
        let resumen_datos = [];
        
        filas.forEach((fila, index) => {
            // Buscamos el botón de eliminar para sacar el ID real
            const btnEliminar = fila.querySelector('.btn-eliminar');
            const idReal = btnEliminar ? parseInt(btnEliminar.dataset.id) : null;
            
            // Texto de la ruta (columna 2)
            const celdaRuta = fila.cells[1]; 
            const textoRuta = celdaRuta ? celdaRuta.innerText : "Ruta desconocida";

            if (idReal) {
                ids_en_pantalla.push(idReal);
                // "Ruta 1 (ID 55): Madrid -> Barcelona..."
                resumen_datos.push(`Ruta visual #${index + 1} (ID BD: ${idReal}): ${textoRuta}`);
            }
        });

        return {
            pagina: "RutasRecientes",
            ids_rutas: ids_en_pantalla, // Lista de ints para que la IA sepa cual borrar
            datos: resumen_datos        // Texto descriptivo para el prompt
        };
    }

    // Enviar mensaje
    async function enviarMensajeChat() {
        const mensaje = chatInput.value.trim();
        if (!mensaje) return;

        // 1. Mostrar mensaje del usuario
        agregarMensajeUI(mensaje, 'user-message');
        chatInput.value = '';

        // 2. Obtener contexto visual
        const contexto = obtenerContextoVisual();

        // 3. Mostrar "Escribiendo..." (opcional)
        const loadingMsgId = agregarMensajeUI("Pensando...", 'bot-message');

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mensaje: mensaje,
                    contexto: contexto
                })
            });

            const data = await response.json();
            
            // Eliminar mensaje de carga
            const loadingElement = document.getElementById(loadingMsgId);
            if(loadingElement) loadingElement.remove();

            if (data.status === 'ok') {
                agregarMensajeUI(data.respuesta, 'bot-message');
                
                // Si la IA borró algo, recargar la tabla automáticamente
                if (data.respuesta.includes("eliminada correctamente")) {
                    await cargarRutasUsuario();
                }
            } else {
                agregarMensajeUI("Error: " + data.message, 'bot-message');
            }

        } catch (error) {
            console.error(error);
            const loadingElement = document.getElementById(loadingMsgId);
            if(loadingElement) loadingElement.remove();
            agregarMensajeUI("Error de conexión con el agente.", 'bot-message');
        }
    }

    function agregarMensajeUI(texto, clase) {
        const div = document.createElement('div');
        div.classList.add('message', clase);
        div.innerText = texto;
        // ID temporal para poder borrarlo si es loading
        const id = 'msg-' + Date.now();
        div.id = id;
        
        messagesContainer.appendChild(div);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        return id;
    }

    sendChatBtn.addEventListener('click', enviarMensajeChat);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') enviarMensajeChat();
    });

/* ########################################################################################################################### */ 
async function cargarRutasUsuario() {
    const loadingDiv = document.getElementById('loading-rutas');
    const noRutasDiv = document.getElementById('no-rutas');
    const tablaRutas = document.getElementById('tabla-rutas');
    const tbody = document.getElementById('tabla-rutas-body');
    const btnRehacer = document.getElementById("rehacerOptim");

    loadingDiv.style.display = 'block';
    noRutasDiv.style.display = 'none';
    tablaRutas.style.display = 'none';
    btnRehacer.disabled = true; // Deshabilitar botón inicialmente

    try {
        const response = await fetch('/obtener_rutas');
        const data = await response.json();

        loadingDiv.style.display = 'none';

        if (data.status === "ok") {
            if (data.rutas.length === 0) {
                noRutasDiv.style.display = 'block';
            } else {
                tablaRutas.style.display = 'table';
                tbody.innerHTML = "";

                data.rutas.forEach((ruta, index) => {
                    const row = document.createElement('tr');

                    row.innerHTML = `
                        <td class="columna-indice">${index + 1}</td>
                        <td>${ruta.ruta_corta}</td>
                        <td style="text-align:center;">
                            <button class="btn-mas-detalles" data-ruta-completa="${ruta.ruta_completa}">
                                Ver más
                            </button>
                        </td>
                        <td style="text-align:center;">
                            <button class="btn-eliminar" data-id="${ruta.id_ruta}" title="Eliminar ruta">
                                Eliminar
                            </button>
                        </td>
                        <td style="text-align:center;">
                            <input type="radio" name="seleccionarRuta" value="${ruta.id_ruta}">
                        </td>
                    `;
                    tbody.appendChild(row);
                });
                const radios = document.querySelectorAll('input[name="seleccionarRuta"]');
                radios.forEach(r => {
                    r.addEventListener('change', () => {
                        btnRehacer.disabled = false;
                    });
                });
            }
        } else {
            noRutasDiv.innerHTML = `<p>Error: ${data.message}</p>`;
            noRutasDiv.style.display = 'block';
        }
    } catch (err) {
        console.error("Error al cargar rutas:", err);
        loadingDiv.style.display = 'none';
        noRutasDiv.innerHTML = `<p>Error al conectar con el servidor.</p>`;
        noRutasDiv.style.display = 'block';
    }
}
