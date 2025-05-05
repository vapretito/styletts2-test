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
    texto = event["input"].get("text", "Hola desde StyleTTS2")
    salida = "/tmp/salida.wav"

    try:
        print("📥 Texto recibido:", texto)

        print("🎤 Generando wav...")
        wav = synthesizer.synthesize(texto, alpha=1.0)
        print("✅ WAV generado. Guardando archivo...")

        torchaudio.save(salida, wav.unsqueeze(0), 24000)

        print("📤 Codificando en base64...")
        with open(salida, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        print("✅ Listo para enviar.")
        return { "audio_base64": audio_base64 }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return { "error": str(e) }

runpod.serverless.start({"handler": handler, "setup": setup})

runpod.serverless.start({"handler": handler, "setup": setup})
