from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import os

# Importando a classe de inferência do nosso script principal
from thyroscan_v25 import ThyroInference

app = FastAPI(title="ThyroScan Pro API", version="25.0")

# Configurar CORS para permitir que o HTML local acesse a API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite qualquer origem para facilitar testes locais
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carregar o modelo globalmente
MODEL_PATH = 'thyroscan_v25_model.pkl'
inference_engine = None

@app.on_event("startup")
def load_model():
    global inference_engine
    if os.path.exists(MODEL_PATH):
        try:
            inference_engine = ThyroInference(MODEL_PATH)
            print(f"✅ Modelo {MODEL_PATH} carregado com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo: {e}")
    else:
        print(f"⚠️ Aviso: O modelo {MODEL_PATH} não foi encontrado.")
        print("   Por favor, treine o modelo primeiro rodando thyroscan_v25.py")

class ImageRequest(BaseModel):
    imagem_base64: str

@app.post("/api/classificar")
def classificar_imagem(request: ImageRequest):
    if not inference_engine:
        raise HTTPException(status_code=503, detail="O modelo não está carregado. Treine o modelo primeiro.")

    try:
        # Remover o prefixo 'data:image/jpeg;base64,' ou similar
        img_str = request.imagem_base64
        if ',' in img_str:
            img_str = img_str.split(',')[1]

        # Decodificar Base64 para Imagem OpenCV
        img_bytes = base64.b64decode(img_str)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)

        if img_cv2 is None:
            raise HTTPException(status_code=400, detail="Imagem inválida ou corrompida.")

        # Realizar a inferência
        classe, prob, roi = inference_engine.classificar_array(img_cv2)

        # Codificar a ROI (Região de Interesse) de volta para Base64 para mostrar no HTML
        _, roi_encoded = cv2.imencode('.png', roi)
        roi_base64 = base64.b64encode(roi_encoded).decode('utf-8')

        return {
            "classe": classe,
            "probabilidade": float(prob),
            "threshold": float(inference_engine.threshold),
            "roi_base64": roi_base64,
            "simulado": False
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Iniciando Servidor Local ThyroScan Pro...")
    print("   Abra o arquivo thyroscan_frontend.html no seu navegador!")
    uvicorn.run(app, host="127.0.0.1", port=8000)
