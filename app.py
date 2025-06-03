from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
import os
from PIL import Image
import gdown

app = Flask(__name__)

MODELS_ENV_VARS = {
    "MobileNet": {"env_var": "MOBILENET_FILE_ID", "filename": "mobilenet_best_model.h5"},
    "MobileNetV2": {"env_var": "MOBILENETV2_FILE_ID", "filename": "mobilenetv2_best_model.h5"},
    "EfficientNetB0": {"env_var": "EFFICIENTNETB0_FILE_ID", "filename": "efficientnetb0_best_model.h5"},
    "EfficientNetV2B0": {"env_var": "EFFICIENTNETV2B0_FILE_ID", "filename": "efficientnetv2b0_best_model.h5"}
}

CLASS_NAMES = {
    0: 'COVID-19',
    1: 'Normal',
    2: 'Pneumonia-Bacterial',
    3: 'Pneumonia-Viral'
}

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_and_load_model(model_key):
    model_config = MODELS_ENV_VARS.get(model_key)
    if not model_config:
        raise FileNotFoundError(f"Konfigurasi model untuk '{model_key}' tidak ditemukan.")

    model_filename = model_config["filename"]
    env_var_name = model_config["env_var"]
    
    model_gdrive_id = os.environ.get(env_var_name)

    local_model_path = os.path.join(MODELS_DIR, model_filename)

    if not os.path.exists(local_model_path):
        print(f"Mengunduh model {model_key} dari Google Drive (ID dari env var: {env_var_name})...")
        try:
            gdown.download(id=model_gdrive_id, output=local_model_path, quiet=False)
            print(f"Model {model_key} berhasil diunduh ke {local_model_path}")
        except Exception as e:
            if os.path.exists(local_model_path):
                try:
                    os.remove(local_model_path)
                except OSError as oe:
                    print(f"Gagal menghapus file parsial {local_model_path}: {oe}")

    if not os.path.exists(local_model_path): 
        raise FileNotFoundError(f"File model {local_model_path} tidak ditemukan setelah mencoba mengunduh.")
        
    print(f"Memuat model {model_key} dari {local_model_path}...")
    return load_model(local_model_path)

def prepare_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    error_message = None
    image_path_display = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename:
            upload_folder = 'static/uploads'
            os.makedirs(upload_folder, exist_ok=True)

            filepath = os.path.join(upload_folder, file.filename)
            
            file.save(filepath)
            image_path_display = filepath

            try:
                img_array = prepare_image(filepath)

                for model_key in MODELS_ENV_VARS.keys():
                    try:
                        print(f"Memproses model: {model_key}")
                        model = download_and_load_model(model_key)
                        preds_raw = model.predict(img_array)
                        confidence = np.max(preds_raw)
                        predicted_class_idx = np.argmax(preds_raw)
                        predicted_class_name = CLASS_NAMES.get(predicted_class_idx, "Kelas Tidak Dikenal")

                        predictions[model_key] = {
                            "class": predicted_class_name,
                            "confidence": f"{confidence:.2%}"
                        }
                        print(f"Prediksi untuk {model_key}: {predictions[model_key]}")
                    except (FileNotFoundError, ValueError, RuntimeError) as model_load_e:
                        print(f"Error dengan model {model_key}: {model_load_e}")
                        predictions[model_key] = str(model_load_e)
                        continue 
                    except Exception as e_inner:
                        print(f"Error saat prediksi dengan model {model_key}: {e_inner}")
                        predictions[model_key] = f"Error prediksi: {str(e_inner)}"


            except Exception as e_outer:
                print(f"Error utama: {e_outer}")
                error_message = f"Error: {str(e_outer)}"
        else:
            error_message = "Tidak ada file yang dipilih atau file tidak valid."

    return render_template('index.html', predictions=predictions, image_path=image_path_display, error=error_message)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host='0.0.0.0', port=port, debug=True)