import pandas as pd
import torch 
from torch import nn
from torch.utils.data import DataLoader
import argparse
# with torch.amp.autocast('cpu', dtype=torch.bfloat16):

# INIT parser
parser = argparse.ArgumentParser()
# Add argument 
parser.add_argument(
    "-e", "--epoch",
    type=int,
    default=30,
    help="Masukin jumlah Epoch!"
)
parser.add_argument(
    "-b", "--batch",
    type=int,
    default=128,
    help="Masukin jumlah Batch!"
)
# Parse Argumentnya
args = parser.parse_args()

# CONSTANT VARIABLE
EPOCH_SIZE = args.epoch
BATCH_SIZE = args.batch
SEED = 42
MONDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv"
TUESDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"
THURSDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
WEDNESDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"
FRIDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours.pcap_ISCX-Mix.csv"
fitur = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
]
list_uji = [
    "TUESDAY",
    "WEDNESDAY",
    "THURSDAY",
    "FRIDAY",
]
torch.manual_seed(SEED)

df = pd.read_csv(MONDAY)

# Rename Kolom "Label" yang punya spasi diawal
data = df.rename(columns=lambda nama: nama.strip())[fitur + ["Label"]].copy()
data["Label"] = data["Label"].str.strip()

# Durasi, jumlah paket, dan total byte harus bernilai nol atau positif.
data = data.loc[data[fitur].ge(0).all(axis=1)]

normal = data.loc[data["Label"] == "BENIGN"].sample(n=500000, random_state=SEED)
latih = normal.iloc[:400000]
kalibrasi = normal.iloc[400000:450000]
uji = normal.iloc[450000:500000]

def ke_tensor(tabel):
    nilai = torch.tensor(tabel[fitur].to_numpy(), dtype=torch.float32)
    return torch.log1p(nilai)

# Hitung skala hanya dari data latihan.
x = ke_tensor(latih)
rata = x.mean(dim=0)
simpangan = x.std(dim=0).clamp_min(1e-6)
x_latih = (x - rata) / simpangan


model = nn.Sequential(
    nn.Linear(5, 2),
    nn.ReLU(),
    nn.Linear(2, 5),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()
loader = DataLoader(x_latih, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCH_SIZE):
    for batch in loader:
        rekonstruksi = model(batch)
        loss = mse(rekonstruksi, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

@torch.no_grad()
def skor_per_flow(tabel):
    data_tensor = ke_tensor(tabel)
    data_normalized = (data_tensor - rata) / simpangan
    return (model(data_normalized) - data_normalized).square().mean(dim=1)


def ringkas_a(hasil, ambang_per_persentil):
    """Hitung target dan FPR aktual pada flow uji berlabel BENIGN."""
    skor_benign = hasil.loc[hasil["Label"] == "BENIGN", "skor"]
    n_benign = len(skor_benign)
    if n_benign == 0:
        raise ValueError("Evaluasi FPR memerlukan flow uji berlabel BENIGN.")

    baris = []
    for q, ambang_q in ambang_per_persentil.items():
        fp = int((skor_benign > ambang_q).sum())
        fpr_persen = 100 * fp / n_benign
        target_persen = 100 - q
        baris.append({
            "hari": "Senin",
            "seed": SEED,
            "persentil": q,
            "ambang_mse": ambang_q,
            "n_benign": n_benign,
            "fp": fp,
            "tn": n_benign - fp,
            "target_persen": target_persen,
            "fpr_persen": fpr_persen,
            "selisih_poin_persen": fpr_persen - target_persen,
            "rasio": fpr_persen / target_persen,
        })
    return pd.DataFrame(baris)


model.eval()
skor_kalibrasi = skor_per_flow(kalibrasi)
skor_uji = skor_per_flow(uji)
if not torch.isfinite(skor_kalibrasi).all() or not torch.isfinite(skor_uji).all():
    raise ValueError("Periksa fitur dan model karena skor kalibrasi atau uji tidak terbatas.")

# Semua ambang berasal dari 50.000 flow kalibrasi yang sama.
ambang_per_persentil = {
    q: torch.quantile(skor_kalibrasi, q / 100).item()
    for q in (90, 95, 97.5, 99, 99.5)
}
ambang = ambang_per_persentil[99]
hasil = uji[["Label"]].copy()
hasil["skor"] = skor_uji.numpy()
hasil["alarm"] = hasil["skor"] > ambang

print(f"Ambang P99: {ambang:.4f}")
print(pd.crosstab(hasil["Label"], hasil["alarm"]))
print(hasil.groupby("Label")["alarm"].mean().mul(100).round(2))

tabel_a = ringkas_a(hasil, ambang_per_persentil)
print(tabel_a.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
tabel_a.to_csv(f"result/hasil_a_senin.csv", index=False)


# ==========================================================================================
# UNCOMMENT BUAT SAVE MODEL
# torch.save({
#     "state_dict": model.state_dict(),
#     "arsitektur": [5, 2, 5],
#     "fitur": fitur,
#     "rata": rata,
#     "simpangan": simpangan,
#     "ambang_per_persentil": ambang_per_persentil,
# }, "checkpoint_senin.pt")
