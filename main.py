from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="API Orientamento - Clustering")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica il modello di clustering
model = joblib.load("modello_clustering.pkl")

# Mappiamo i numeri dei cluster a nomi comprensibili (da aggiustare dopo aver visto i cluster)
cluster_names = {
    0: "Scientifico-Tecnologico",
    1: "Umanistico-Creativo",
    2: "Economico-Pratico"
}

class FormData(BaseModel):
    q1_problem_solving: int
    q2_scienze: int
    q3_umanistiche: int
    q4_tecnologia: int
    q5_economia: int
    q6_social_impact: int
    q7_creativita: int
    q8_sicurezza: int
    q9_esperienze_pratiche: int

@app.get("/")
def root():
    return {"status": "API Clustering attiva ✅", "type": "KMeans (non supervisionato)"}

@app.post("/predict")
def predict(data: FormData):
    try:
        df_input = pd.DataFrame([data.dict()])
        cluster_id = model.predict(df_input)[0]

        nome_area = cluster_names.get(cluster_id, f"Cluster {cluster_id}")

        if data.q8_sicurezza >= 4:
            livello = "alto"
            consiglio = f"Ti consigliamo l'area **{nome_area}**! Sembra molto in linea con i tuoi interessi."
        elif data.q8_sicurezza >= 2:
            livello = "medio"
            consiglio = f"Ti consigliamo l'area **{nome_area}**. Potrebbe essere una buona direzione da esplorare."
        else:
            livello = "da esplorare"
            consiglio = f"Risultato: **{nome_area}**. Ti consigliamo di parlare con un orientatore per confermare."

        return {
            "cluster": int(cluster_id),
            "area": nome_area,
            "livello_orientamento": livello,
            "consiglio": consiglio
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
