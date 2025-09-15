from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import io

app = FastAPI()

@app.post("/extract-colors/")
async def extract_colors(file: UploadFile = File(...), num_colors: int = 5):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((200, 200))
    pixels = np.array(img).reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)

    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_idx = np.argsort(-counts)

    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(*colors[i]) for i in sorted_idx
    ]

    return JSONResponse({"colors": hex_colors})
