import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from torchvision.utils import make_grid

# ===========================
#   Modelo: DCGAN Generator
# ===========================

class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature=64, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*8, feature*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*4, feature*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature*2, feature, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z)


# ===========================
# Configuración
# ===========================

latent_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "exp1_base_G.pth"

G = Generator(latent_dim=latent_dim, feature=64).to(device)
G.load_state_dict(torch.load(model_path, map_location=device))
G.eval()


# ===========================
# Función para generar logo
# ===========================
def generar_logo():
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        fake = G(noise).cpu()
    img = fake.squeeze(0)
    img = (img * 0.5 + 0.5).numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


# ===========================
# UI — Interfaz bonita
# ===========================

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
        img_output = gr.Image(label="Logo generado", height=256, width=256)

    btn = gr.Button("🎨 Generar nuevo logo", elem_id="button")
    btn.click(fn=generar_logo, outputs=img_output)

demo.launch()
