import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.metrics.pairwise import cosine_similarity

# Cargar tu dataset de películas (asegúrate de que la ruta sea correcta)
df = pd.read_csv('GenerosPorPeliculaConTitulo.csv')  # Cambia esto a la ruta de tu dataset

# Inicializa la aplicación FastAPI
app = FastAPI()

def recomendar_peliculas_por_titulo(title: str, df: pd.DataFrame, top_n: int = 5) -> list:
    # Filtrar la película seleccionada por título
    pelicula_seleccionada = df[df['title'] == title]
    
    if pelicula_seleccionada.empty:
        raise HTTPException(status_code=404, detail="La película no se encuentra en el dataset.")

    # Obtener las características de la película seleccionada
    pelicula_vector = pelicula_seleccionada.iloc[:, 2:].values  # Excluir el movie_id y el title
    similares = cosine_similarity(pelicula_vector, df.iloc[:, 2:])  # Calcular similitudes

    # Crear un DataFrame de similitudes
    similares_df = pd.DataFrame(similares.T, index=df['title'], columns=['similarity'])
    
    # Obtener las top_n películas más similares
    recomendaciones = similares_df.sort_values(by='similarity', ascending=False).head(top_n + 1)  # +1 para excluir la película seleccionada
    recomendaciones = recomendaciones[recomendaciones.index != title]  # Excluir la película seleccionada

    # Obtener la lista de recomendaciones
    peliculas_recomendadas = recomendaciones.index.tolist()

    return peliculas_recomendadas  # Devolver la lista de películas recomendadas

@app.get("/")
async def welcome():
    return {
        "message": "Bienvenido a la API de recomendación de películas.",
        "funcionalidad": "Esta API te permite obtener recomendaciones de películas basadas en una película que ya conoces.",
        "como_usar": "Envía una solicitud GET a /recomendar/ con el título de la película como un parámetro de consulta.",
        "ejemplo": {
            "url": "http://localhost:8000/recomendar/?title=Inception&top_n=5",
            "nota": "Reemplaza 'Inception' con el título de la película que deseas."
        }
    }

@app.get("/recomendar/")
async def recomendar_movies(title: str, top_n: int = 5):
    recomendaciones = recomendar_peliculas_por_titulo(title, df, top_n)
    return {"recomendaciones": recomendaciones}

# Para correr la aplicación, usa el siguiente comando en la terminal:
# uvicorn main:app --reload
