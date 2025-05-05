import runpod
import torch
import base64
import os
import torchaudio

# Importar las clases de StyleTTS2
from StyleTTS.utils.tools import load_config
from StyleTTS.inference.synthesis import StyleTTS_Synthesizer

# Cargar modelo en setup
synthesizer = None

def setup():
    global synthesizer
    print("🌀 Cargando StyleTTS2...")

    # Ruta a la configuración y los modelos preentrenados
    config_path = "/workspace/StyleTTS/Configs/config.yml"
    ckpt_1st = "/workspace/StyleTTS/logs/epoch_1st_00499.pth"
    ckpt_2nd = "/workspace/StyleTTS/logs/epoch_2nd_00499.pth"

    config = load_config(config_path)

    synthesizer = StyleTTS_Synthesizer(
        config=config,
        ckpt_1st=ckpt_1st,
        ckpt_2nd=ckpt_2nd,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print("✅ StyleTTS2 cargado y listo.")

def handler(event):
    global synthesizer
    text = event["input"].get("text", "Texto vacío")

    # Salida temporal
    output_path = "/tmp/salida.wav"

    try:
        # Inferencia
        wav = synthesizer.synthesize(text, alpha=1.0)
        torchaudio.save(output_path, wav.unsqueeze(0), 24000)

        # Codificar a base64
        with open(output_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return {"audio_base64": audio_base64}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler, "setup": setup})
