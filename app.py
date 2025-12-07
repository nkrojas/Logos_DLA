import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
import gradio as gr

# ----------------------
# Modelo DCGAN (Generator)
# ----------------------
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

# ----------------------
# Cargar modelo entrenado
# ----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100  # el del experimento 1
feature = 64

G = Generator(latent_dim=latent_dim, feature=feature).to(device)
G.load_state_dict(torch.load("exp1_base_G.pth", map_location=device))
G.eval()


# ----------------------
# Función: generar 1 logo
# ----------------------
def generar_logo():
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, 1, 1, device=device)
        img = G(noise).cpu()[0]

    img = (img * 0.5 + 0.5).clamp(0, 1)  # desnormalizar
    img = img.permute(1, 2, 0).numpy()  # CHW → HWC
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)

# ----------------------
# Interfaz Gradio
# ----------------------
demo = gr.Interface(
    fn=generar_logo,
    inputs=None,
    outputs=gr.Image(label="Logo generado"),
    title="Generador de Logos – DCGAN Exp1",
    description="Haz clic para generar un logo distinto cada vez."
)

demo.launch()
