import torch
import gradio as gr
import numpy as np
from torchvision.utils import make_grid
from model import Generator  # si está en otro archivo, ajusta
# 🔥 Ajusta el nombre del archivo de tu modelo
MODEL_PATH = "exp1_base_G.pth"

# ===================
# Configuración
# ===================
latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo
G = Generator(latent_dim=latent_dim, feature=64).to(device)
G.load_state_dict(torch.load(MODEL_PATH, map_location=device))
G.eval()


# ===================
# Generador de logo
# ===================
def generar_logo():
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake = G(noise).cpu()
    img = fake.squeeze(0)
    img = (img * 0.5 + 0.5).numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


# ===================
# Interfaz Gradio
# ===================
css = """
#title {
  text-align: center;
  font-size: 34px !important;
  font-weight: 700;
  color: #2A2A2A;
  margin-bottom: 20px;
}
#subtitle {
  text-align: center;
  font-size: 18px;
  color: #555;
  margin-bottom: 25px;
}
#button {
  background: #2A7FFF !important;
  color: white !important;
  font-size: 18px !important;
  border-radius: 10px !important;
  height: 60px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("<div id='title'>🪄 Generador de Logos con DCGAN</div>")
    gr.Markdown("<div id='subtitle'>Proyecto de IA Generativa — Diseño de logotipos abstractos</div>")

    with gr.Row():
        img_output = gr.Image(label="Logo generado", height=256, width=256, elem_id="preview")

    btn = gr.Button("🎨 Generar nuevo logo", elem_id="button")
    btn.click(fn=generar_logo, outputs=img_output)

demo.launch()
