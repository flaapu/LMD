# 1. Imagen base oficial de Python ligera
FROM python:3.12-slim

# 2. Configurar directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar dependencias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<70.0.0" && pip install --no-cache-dir -r requirements.txt

# 4. Copiar todo el contenido del proyecto
COPY . .

# 5. PASO CRUCIAL: Ejecutar los scripts de Machine Learning
RUN python scr/data.py && python scr/train.py

# 6. PASO DE CONTROL: Ejecutar Pytest inyectando la ruta raíz de Python
RUN PYTHONPATH=. pytest

# 7. Exponer el puerto de FastAPI
EXPOSE 8000

# 8. Comando para levantar el servidor de FastAPI usando Uvicorn
CMD ["uvicorn", "scr.app:app", "--host", "0.0.0.0", "--port", "8000"]