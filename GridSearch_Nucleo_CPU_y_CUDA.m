clear; clc; close all;

% Solicitar al usuario el número de veces que se debe repetir el algoritmo
Num_iteraciones = 10;

% Leer el archivo CSV
data = readmatrix('distancias.csv');
Distancias = gpuArray(data(:, 2:end)); % Transferir a GPU
DistanciasGPU = Distancias; % Renombrar para claridad

% Definir rangos de parámetros para Grid Search
param_ranges = {
    'Num_Pob', [20, 50, 100];
    'Num_Gen', [50, 100, 200];
    'Pm', [0.05, 0.1, 0.2];
    'm', [2, 3, 5];
    'Num_Competidores', [2, 3, 5];
    'Hijos_Crossover', [1, 2];
};

% Generar combinaciones de parámetros
combinations = combvec(param_ranges{:, 2});
num_combinations = size(combinations, 2);

% Inicializar resultados
Resultados_GridSearch = cell(num_combinations, 11);

% Configurar el uso de 10 núcleos
%parpool(10); % Inicializar el pool de procesamiento paralelo

for c = 1:num_combinations
    fprintf('%d / %d\n', c, num_combinations);
    % Extraer configuración actual
    params = combinations(:, c);

    % Asignar valores de parámetros
    Num_Pob = params(1);
    Num_Gen = params(2);
    Pm = params(3);
    m = params(4);
    Num_Competidores = params(5);
    Hijos_Crossover = params(6);

    % Ejecutar el algoritmo con la configuración actual
    [mejor, media, peor, desviacion, tiempo] = EjecutarAlgoritmo(Num_Pob, Num_Gen, Pm, m, Num_Competidores, Hijos_Crossover, DistanciasGPU, Num_iteraciones);

    % Guardar resultados
    Resultados_GridSearch(c, :) = {Num_Pob, Num_Gen, Pm, m, Num_Competidores, Hijos_Crossover, mejor, media, peor, desviacion, tiempo};
end

% Eliminar el pool de procesamiento paralelo
delete(gcp('nocreate'));

%% Análisis de los resultados
% Convertir los resultados a una tabla
T = cell2table(Resultados_GridSearch, 'VariableNames', {'Num_Pob', 'Num_Gen', 'Pm', 'm', 'Num_Competidores', 'Hijos_Crossover', 'Mejor', 'Media', 'Peor', 'Desviacion', 'Tiempo'});

% 1. Menor tiempo
mejor_tiempo = sortrows(T, 'Tiempo');
mejor_tiempo = mejor_tiempo(1:5, :);

% 2. Menor valor de "Mejor" (ahora buscamos minimizar esta métrica)
mejor_valor = sortrows(T, 'Mejor');
mejor_valor = mejor_valor(1:5, :);

% 3. Menor desviación estándar
menor_desviacion = sortrows(T, 'Desviacion');
menor_desviacion = menor_desviacion(1:5, :);

% 4. Combinación ponderada: tiempo, "Mejor", y desviación
% Normalización de las métricas para crear una métrica combinada
T.Metrica_Compuesta = ...
    -T.Mejor / max(T.Mejor) - ... % Invertir porque queremos minimizar
    T.Tiempo / max(T.Tiempo) - ...
    T.Desviacion / max(T.Desviacion);

% Seleccionar las mejores combinaciones según esta métrica
mejor_combinada = sortrows(T, 'Metrica_Compuesta', 'descend');
mejor_combinada = mejor_combinada(1:5, :);

%% Mostrar resultados
disp('Menor Tiempo:');
disp(mejor_tiempo);

disp('Menor Valor de ''Mejor'':');
disp(mejor_valor);

disp('Menor Desviación:');
disp(menor_desviacion);

disp('Métrica Combinada:');
disp(mejor_combinada);


%% Función principal del algoritmo genético con CUDA
function [mejor, media, peor, desviacion, tiempo] = EjecutarAlgoritmo(Num_Pob, Num_Gen, Pm, m, Num_Competidores, Hijos_Crossover, DistanciasGPU, Num_iteraciones)

tic; % Iniciar el temporizador
Resultados_Generales = gpuArray.zeros(1, Num_iteraciones);

Num_var = size(DistanciasGPU, 1);  % Número de ciudades basado en la matriz de distancias
parfor iteracion = 1:Num_iteraciones
    % Población inicial en GPU
    Pob = gpuArray.zeros(Num_Pob, Num_var);
    for i = 1:Num_Pob
        Pob(i, :) = randperm(Num_var, 'gpuArray');
        Pob(i, :) = heuristica_abruptosGPU(Pob(i, :), m, DistanciasGPU, Num_var);
    end

    % Evaluación inicial
    Aptitud_Pob = arrayfun(@(row) CostoGPU(Pob(row, :), DistanciasGPU, Num_var), 1:Num_Pob);

    % Inicializar mejor histórico
    [Mejor_Aptitud_Historico, idx] = min(Aptitud_Pob);
    Mejor_Individuo_Historico = Pob(idx, :);

    iter = 1;
    while iter <= Num_Gen
        % Selección de Padres por torneo
        Padres = SeleccionTorneoGPU(Pob, Aptitud_Pob, Num_Pob, Num_Competidores);

        % Cruzamiento y evaluación
        Hijos = gpuArray.zeros(Num_Pob, Num_var);
        Aptitud_Hijos = gpuArray.zeros(Num_Pob, 1);

        for i = 1:2:Num_Pob
            if Hijos_Crossover == 1
                % Generar hijo 1
                hijo_1 = CycleCrossoverGPU(Padres(i, :), Padres(i+1, :), Num_var);

                % Aplicación de heurística de remoción de abruptos
                hijo_1 = heuristica_abruptosGPU(hijo_1, m, DistanciasGPU, Num_var);

                % Calcular aptitudes de los hijos
                Aptitud_Hijo_1 = CostoGPU(hijo_1, DistanciasGPU, Num_var);

                % Calcular aptitudes de los Padres
                Aptitud_Padre_1 = CostoGPU(Padres(i, :), DistanciasGPU, Num_var);
                Aptitud_Padre_2 = CostoGPU(Padres(i+1, :), DistanciasGPU, Num_var);

                % Crear una matriz con todos los individuos y sus aptitudes
                individuos = gpuArray([Padres(i, :); Padres(i+1, :); hijo_1]);
                aptitudes = gpuArray([Aptitud_Padre_1; Aptitud_Padre_2; Aptitud_Hijo_1]);

                % Ordenar individuos por aptitud
                [aptitudes_ordenadas, indices] = sort(aptitudes);
                mejores_individuos = individuos(indices(1:2), :);
                mejores_aptitudes = aptitudes_ordenadas(1:2);

                % Guardar los mejores individuos y sus aptitudes para la siguiente generación
                Hijos(i, :) = mejores_individuos(1, :);
                Hijos(i+1, :) = mejores_individuos(2, :);
                Aptitud_Hijos(i) = mejores_aptitudes(1);
                Aptitud_Hijos(i+1) = mejores_aptitudes(2);

            else
                % Generar hijo 1 e hijo 2
                hijo_1 = CycleCrossoverGPU(Padres(i, :), Padres(i+1, :), Num_var);
                hijo_2 = CycleCrossoverGPU(Padres(i+1, :), Padres(i, :), Num_var);

                % Aplicación de heurística de remoción de abruptos
                hijo_1 = heuristica_abruptosGPU(hijo_1, m, DistanciasGPU, Num_var);
                hijo_2 = heuristica_abruptosGPU(hijo_2, m, DistanciasGPU, Num_var);

                % Calcular aptitudes de los hijos
                Aptitud_Hijo_1 = CostoGPU(hijo_1, DistanciasGPU, Num_var);
                Aptitud_Hijo_2 = CostoGPU(hijo_2, DistanciasGPU, Num_var);

                % Calcular aptitudes de los Padres
                Aptitud_Padre_1 = CostoGPU(Padres(i, :), DistanciasGPU, Num_var);
                Aptitud_Padre_2 = CostoGPU(Padres(i+1, :), DistanciasGPU, Num_var);

                % Crear una matriz con todos los individuos y sus aptitudes
                individuos = gpuArray([Padres(i, :); Padres(i+1, :); hijo_1; hijo_2]);
                aptitudes = gpuArray([Aptitud_Padre_1; Aptitud_Padre_2; Aptitud_Hijo_1; Aptitud_Hijo_2]);

                % Ordenar individuos por aptitud
                [aptitudes_ordenadas, indices] = sort(aptitudes);
                mejores_individuos = individuos(indices(1:2), :);
                mejores_aptitudes = aptitudes_ordenadas(1:2);

                % Guardar los mejores individuos y sus aptitudes para la siguiente generación
                Hijos(i, :) = mejores_individuos(1, :);
                Hijos(i+1, :) = mejores_individuos(2, :);
                Aptitud_Hijos(i) = mejores_aptitudes(1);
                Aptitud_Hijos(i+1) = mejores_aptitudes(2);
            end
        end

        % Reemplazar nueva población con hijos
        Pob = Hijos;
        Aptitud_Pob = Aptitud_Hijos;

        % Mutación
        for i = 1:Num_Pob
            if rand <= Pm
                Pob(i, :) = MutacionGPU(Pob(i, :), Num_var);
                Aptitud_Pob(i) = CostoGPU(Pob(i, :), DistanciasGPU, Num_var);
            end
        end


        % Actualizar mejor histórico
        [Mejor_Aptitud_Generacion, idx] = min(Aptitud_Pob);
        if Mejor_Aptitud_Generacion < Mejor_Aptitud_Historico
            Mejor_Aptitud_Historico = Mejor_Aptitud_Generacion;
            Mejor_Individuo_Historico = Pob(idx, :);
        end
        iter = iter + 1;
    end

    Resultados_Generales(iteracion) = Mejor_Aptitud_Historico;
end

% Estadísticas finales
mejor = min(Resultados_Generales);
media = mean(Resultados_Generales);
peor = max(Resultados_Generales);
desviacion = std(Resultados_Generales);
tiempo = toc; % Terminar el temporizador
end

function hijo_premium = heuristica_abruptosGPU(Hijo, m, distanciasGPU, num_ciudades)
    for i = 1:num_ciudades
        % Selección entre ciudades cercanas
        [~, idx] = sort(distanciasGPU(i, :)); % Ordenar por distancia
        idx = idx(2:m+1); % Tomar las m ciudades más cercanas
        idx = randsample(idx, 1, true); % Seleccionar una al azar

        % Posición de inserción
        posiciones = find(Hijo == idx); % Encontrar posiciones del nodo
        posiciones = [posiciones, posiciones + 1];

        % Eliminar ciudad de su posición
        Ruta = Hijo;
        Premove = find(Hijo == i);
        Ruta(Premove) = [];

        % Ajustar las posiciones de inserción según la eliminación
        posiciones(posiciones > Premove) = posiciones(posiciones > Premove) - 1;

        % Insertar el elemento en las nuevas posiciones (Concatenación)
        Ruta1 = [Ruta(1:posiciones(1)-1), i, Ruta(posiciones(1):end)];
        Ruta2 = [Ruta(1:posiciones(2)-1), i, Ruta(posiciones(2):end)];

        % Seleccionar la mejor ruta entre: Hijo, Ruta1 y Ruta2
        Aptitud_Hijo = CostoGPU(Hijo, distanciasGPU, num_ciudades);
        Ruta_1 = CostoGPU(Ruta1, distanciasGPU, num_ciudades);
        Ruta_2 = CostoGPU(Ruta2, distanciasGPU, num_ciudades);

        % Sustitución
        individuos = gpuArray([Hijo; Ruta1; Ruta2]);
        aptitudes = gpuArray([Aptitud_Hijo; Ruta_1; Ruta_2]);
        [~, idx] = sort(aptitudes); % Ordenar por aptitud
        Hijo = individuos(idx(1), :); % Seleccionar el mejor
    end
    hijo_premium = Hijo;
end

function hijo = CycleCrossoverGPU(padre1, padre2, noCiudades)
    hijo = gpuArray.zeros(1, noCiudades);          % Inicializar el hijo con ceros
    visitado = gpuArray.false(1, noCiudades);      % Marcadores de las posiciones visitadas
    ciclo = 0;                                    % Contador para el ciclo

    while any(~visitado) % Mientras haya alguna ciudad no visitada
        pos_inicio = find(~visitado, 1);          % Primera ciudad no visitada
        ciclo = ciclo + 1;
        pos_actual = pos_inicio;

        % Alternar Padres según el ciclo
        if mod(ciclo, 2) == 1
            padre_actual = padre1;
            otro_padre = padre2;
        else
            padre_actual = padre2;
            otro_padre = padre1;
        end

        while true
            hijo(pos_actual) = padre_actual(pos_actual); % Asignar valor
            visitado(pos_actual) = true;                % Marcar como visitado

            % Buscar siguiente posición
            valor_actual = padre_actual(pos_actual);
            pos_actual = find(otro_padre == valor_actual, 1);

            % Validar posición actual
            if isempty(pos_actual)
                error('Error: posición actual no encontrada en el otro padre.');
            end

            % Condición de salida del ciclo
            if visitado(pos_actual) || pos_actual == pos_inicio
                break;
            end
        end
    end
end

function costo = CostoGPU(recorrido, DistanciasGPU, noCiudades)
    % Desplazar el recorrido circularmente
    recorridoShifted = circshift(recorrido, -1);
    % Calcular índices de las distancias
    indices = sub2ind(size(DistanciasGPU), recorrido, recorridoShifted);
    % Sumar las distancias
    costo = sum(DistanciasGPU(indices));
end

function mutado = MutacionGPU(individuo, noCiudades)
    idx = randperm(noCiudades, 2, 'gpuArray'); % Seleccionar dos posiciones aleatorias en la GPU
    mutado = individuo;
    % Intercambiar los valores en las posiciones seleccionadas
    mutado([idx(1), idx(2)]) = mutado([idx(2), idx(1)]);
end

function Padres = SeleccionTorneoGPU(Pob, Aptitud_Pob, Num_Pob, Num_Competidores)
    Padres = gpuArray.zeros(Num_Pob, size(Pob, 2)); % Inicializar matriz de padres
    for i = 1:Num_Pob
        Competidores = randperm(Num_Pob, Num_Competidores, 'gpuArray'); % Seleccionar competidores aleatorios
        [~, idx] = min(Aptitud_Pob(Competidores)); % Encontrar el índice del mejor competidor
        Padres(i, :) = Pob(Competidores(idx), :);  % Asignar al padre seleccionado
    end
end

