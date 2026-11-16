"""Demo autoencoder CPU dengan dua fitur trafik sintetis.
Jalankan: python prototipe_autoencoder.py
"""
import torch
from torch import nn

torch.manual_seed(42)


def buat_flow_normal(jumlah):
    paket = torch.randint(10, 51, (jumlah,)).float()
    byte_per_paket = 500 + 15 * torch.randn(jumlah)
    return torch.stack((paket, (paket * byte_per_paket).round()), dim=1)


# Satu baris berisi [jumlah paket, total byte].
latih = buat_flow_normal(1000)
kalibrasi = buat_flow_normal(500)
uji = torch.tensor([[30, 15000], [30, 3000]], dtype=torch.float32)

# Samakan skala fitur agar satuan byte tidak mendominasi skor.
rata = latih.mean(dim=0)
simpangan = latih.std(dim=0)
x = (latih - rata) / simpangan

encoder = nn.Linear(2, 1)
decoder = nn.Linear(1, 2)
model = nn.Sequential(encoder, decoder)
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
mse = nn.MSELoss()

# Target rekonstruksi adalah input itu sendiri.
for epoch in range(1, 301):
    rekonstruksi = model(x)
    loss = mse(rekonstruksi, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch in (1, 100, 300):
        print(f"Epoch {epoch:3d} | MSE latih {loss.item():.5f}")


def skor_per_flow(data):
    fitur = (data - rata) / simpangan
    return (model(fitur) - fitur).square().mean(dim=1)


model.eval()
with torch.no_grad():
    ambang = torch.quantile(skor_per_flow(kalibrasi), 0.99)
    print(f"Ambang P99 = {ambang.item():.5f}")
    for flow, skor in zip(uji, skor_per_flow(uji)):
        status = "anomali" if skor > ambang else "normal"
        print(f"{int(flow[0])} paket, {int(flow[1])} byte | skor {skor.item():.5f} | {status}")

