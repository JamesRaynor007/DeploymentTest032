import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from sklearn.metrics.pairwise import cosine_similarity

# Define las rutas para los archivos CSV
file_path = os.path.join(os.path.dirname(__file__), 'GenerosPorPeliculaConTitulo.csv')  # Cambia esto a la ruta de tu dataset

# Cargar tu dataset de películas
df = pd.read_csv(file_path)

# Inicializa la aplicación FastAPI
app = FastAPI()

def recomendar_peliculas_por_titulo(title: str, df: pd.DataFrame, top_n: int = 5) -> list:
    # Convertir el título ingresado a minúsculas
    title_lower = title.lower()
    
    # Filtrar la película seleccionada por título, convirtiendo los títulos en el DataFrame a minúsculas
    pelicula_seleccionada = df[df['title'].str.lower() == title_lower]
    
    if pelicula_seleccionada.empty:
        raise HTTPException(status_code=404, detail="La película no se encuentra en el dataset.")

    # Obtener las características de la película seleccionada
    pelicula_vector = pelicula_seleccionada.iloc[:, 2:].values  # Excluir el movie_id y el title
    similares = cosine_similarity(pelicula_vector, df.iloc[:, 2:])  # Calcular similitudes

    # Crear un DataFrame de similitudes
    similares_df = pd.DataFrame(similares.T, index=df['title'], columns=['similarity'])
    
    # Obtener las top_n películas más similares
    recomendaciones = similares_df.sort_values(by='similarity', ascending=False).head(top_n + 1)  # +1 para excluir la película seleccionada
    recomendaciones = recomendaciones[recomendaciones.index.str.lower() != title_lower]  # Excluir la película seleccionada

    # Obtener la lista de recomendaciones
    peliculas_recomendadas = recomendaciones.index.tolist()

    return peliculas_recomendadas  # Devolver la lista de películas recomendadas

@app.get("/")
async def welcome(request: Request):
    # Obtener la URL base de la solicitud
    base_url = f"{request.url.scheme}://{request.url.hostname}" + (f":{request.url.port}" if request.url.port else "")
    
    return {
        "message": "Bienvenido a la API de recomendación de películas.",
        "funcionalidad": "Esta API te permite obtener recomendaciones de películas basadas en una película que ya conoces.",
        "ejemplo": {
            "url": f"{base_url}/recomendar/?title=Inception",
            "nota": "Reemplaza 'Inception' con el título de la película que conoces y te sugerirá 5 títulos similares. La cantidad de recomendaciones es fija y no puede ser modificada."
        }
    }

@app.get("/recomendar/")
async def recomendar_movies(title: str):
    top_n = 5  # Establecer top_n a 5 y no permitir que sea modificado por el usuario
    recomendaciones = recomendar_peliculas_por_titulo(title, df, top_n)
    return {"recomendaciones": recomendaciones}

# Para correr la aplicación, usa el siguiente comando en la terminal:
# uvicorn main:app --reload
