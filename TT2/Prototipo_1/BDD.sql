-- Tabla "Usuario"
CREATE TABLE Usuario (id_usuario INT AUTO_INCREMENT PRIMARY KEY,
                      nombre VARCHAR(50) NOT NULL,
                      correo VARCHAR(100) NOT NULL UNIQUE,
                      contraseña_hash VARCHAR(255) NOT NULL);

-- Tabla "Ruta"
CREATE TABLE Ruta (id_ruta INT AUTO_INCREMENT PRIMARY KEY,
                   id_usuario INT NOT NULL,
                   destinos JSON NOT NULL,
                   CONSTRAINT fk_ruta_usuario FOREIGN KEY (id_usuario) REFERENCES Usuario(id_usuario));

-- Inserciones de ejemplo en "Usuario"
INSERT INTO Usuario (nombre, correo, contraseña_hash) VALUES
('Ana López', 'ana.lopez@mail.com', '$2b$12$EjemploHash1'),
('Carlos Pérez', 'carlos.perez@mail.com', '$2b$12$EjemploHash2'),
('María García', 'maria.garcia@mail.com', '$2b$12$EjemploHash3'),
('Luis Torres', 'luis.torres@mail.com', '$2b$12$EjemploHash4'),
('Sofía Ramírez', 'sofia.ramirez@mail.com', '$2b$12$EjemploHash5'),
('Miguel Hernández', 'miguel.hernandez@mail.com', '$2b$12$EjemploHash6'),
('Laura Sánchez', 'laura.sanchez@mail.com', '$2b$12$EjemploHash7'),
('Jorge Díaz', 'jorge.diaz@mail.com', '$2b$12$EjemploHash8'),
('Fernanda Cruz', 'fernanda.cruz@mail.com', '$2b$12$EjemploHash9'),
('Ricardo Gómez', 'ricardo.gomez@mail.com', '$2b$12$EjemploHash10');

-- Inserciones de ejemplo en "Ruta"
INSERT INTO Ruta (id_usuario, destinos) VALUES
(1, '[1,44,12,88,137,15]'),
(1, '[22,56,89,103,145]'),
(2, '[5,19,77,130,200,220]'),
(2, '[11,48,92,150]'),
(3, '[33,67,101,140,199]'),
(3, '[9,27,84,138,210]'),
(4, '[12,25,36,99,180,214]'),
(4, '[7,45,88,122,201]'),
(5, '[13,59,78,155,198]'),
(5, '[3,47,100,165,209]'),
(6, '[8,40,72,130,175,220]'),
(6, '[18,54,91,144,200]'),
(7, '[2,28,63,109,150]'),
(7, '[6,39,82,133,177]'),
(8, '[14,46,95,120,165]'),
(8, '[25,69,111,170,210]'),
(9, '[32,74,108,149,202]'),
(9, '[4,50,93,135,190]'),
(10, '[10,42,80,140,185]'),
(10, '[21,66,123,160,219]');
