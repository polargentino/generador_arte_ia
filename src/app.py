import os
import requests
from dotenv import load_dotenv
from PIL import Image
import io
import gradio as gr

# Cargar API key desde .env
load_dotenv()
API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {API_KEY}"}

def generar_imagen(prompt):
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()  # Lanza error si la API falla
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"Error: {e}")
        return None

# Interfaz con Gradio
interfaz = gr.Interface(
    fn=generar_imagen,
    inputs=gr.Textbox(label="Describe la imagen que quieres generar"),
    outputs=gr.Image(label="Imagen generada"),
    title="Generador de Arte con IA 🎨 de Pol Monsalvo",
    description="Usa Stable Diffusion XL para crear imágenes desde texto. Ejemplo: 'un gato astronauta, estilo pixel art'"
)

if __name__ == "__main__":
    interfaz.launch()