from fastapi import FastAPI, UploadFile, File, Request
import os
from PIL import Image
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


import utils as utils
import clip_model as clip

app = FastAPI()
app.mount("/static", StaticFiles(directory="../static2"), name="static")
templates = Jinja2Templates(directory="../templates")


@app.get("/img")
async def img():
    path = "C:/Users/fmolnar/Pictures/Saved Pictures"
    for filename in [filename for filename in os.listdir(path) if filename.endswith(".png") or filename.endswith(".PNG") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        image = Image.open(os.path.join(path, filename)).convert("RGB")
        break
    print(name)
    return FileResponse(os.path.join(path, filename))


@app.post("/upload/")
async def create_upload_file(label_file: UploadFile, image_folder_zip: bytes = File(...)):
    utils.unzip_folder(folder_zip=image_folder_zip, output_path="unzipped")     # unzips the file to folder "unzipped"
    images, names = utils.load_images("unzipped")
    labels = label_file.file.read().decode("utf-8").splitlines()

    top_probs, top_labels = clip.get_n_best_matches(labels, images, n=4)

    results_path = utils.plot_best_predictions(images, labels, top_probs, top_labels)
    print(results_path)
    return FileResponse(results_path, media_type="image/png")

@app.get("/", response_class=HTMLResponse)
async def serve_website(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/cifar/")
async def serve_website():
    return FileResponse("../static2/Cifar100Classes.txt", media_type="text/txt")
