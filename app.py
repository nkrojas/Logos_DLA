import torch
import numpy as np
import gradio as gr
from torchvision.utils import make_grid
from torchvision import transforms
import random
from PIL import Image
import torch.nn as nn

# ====== Modelo ======

class Generator(nn.Module):
    def __init__(self, latent_dim=128, feature=64, channels=3):
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

# ====== Cargar modelo ======

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 128

G = Generator(latent_dim=latent_dim, feature=64).to(device)
G.load_state_dict(torch.load("exp1_base_G.pth", map_location=device))
G.eval()

# ====== Inferencia ======

def generar_logo():
    with torch.no_grad():
        z = torch.randn(1, latent_dim, 1, 1, device=device)
        fake = G(z)[0].cpu()
    # convertir a imagen PIL
        img = (fake + 1) / 2
        img = transforms.ToPILImage()(img)
    return img

# ====== CSS ======

css = """
body {background: #111;}
h1 {text-align:center; color:white;}
button {border-radius: 10px;}
"""

# ====== Interfaz ======

with gr.Blocks() as demo:
    gr.HTML(f"<style>{css}</style>")
    gr.Markdown("# 🎨 Generador de Logos con DCGAN")
    gr.Markdown("Exp1 — Modelo 64x64 entrenado con SimpleIcons")

    output = gr.Image(label="Tu logo generado")

    btn = gr.Button("✨ Nuevo Logo")
    btn.click(fn=generar_logo, outputs=output)

demo.launch()
