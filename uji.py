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
torch.manual_seed(SEED)

DAY = FRIDAY
DAY_NAME = "FRIDAY"
df = pd.read_csv(DAY)

paket = torch.load(
    "model_senin.pt",
    map_location="cpu",
    weights_only=True,
)

model = nn.Sequential(
    nn.Linear(5, 2),
    nn.ReLU(),
    nn.Linear(2, 5),
)

# Isi jaringan dengan bobot hasil training Senin.
model.load_state_dict(paket["state_dict"])
model.eval()

# Ambil kembali fitur, standardisasi, dan ambangnya.
fitur = paket["fitur"]
rata = paket["rata"]
simpangan = paket["simpangan"]
ambang_per_persentil = paket["ambang_per_persentil"]


# ===============================================================================
data = df.rename(columns=lambda nama: nama.strip())[fitur + ["Label"]].copy()
data["Label"] = data["Label"].str.strip()

data = data.loc[data[fitur].ge(0).all(axis=1)]

def ke_tensor(tabel):
    nilai = torch.tensor(tabel[fitur].to_numpy(), dtype=torch.float32)
    return torch.log1p(nilai)

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
            "hari": DAY_NAME,
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

skor_uji = skor_per_flow(data)

ambang = ambang_per_persentil[99]
hasil = data[["Label"]].copy()
hasil["skor"] = skor_uji.numpy()
hasil["alarm"] = hasil["skor"] > ambang

print(f"Ambang P99: {ambang:.4f}")
print(pd.crosstab(hasil["Label"], hasil["alarm"]))
print(hasil.groupby("Label")["alarm"].mean().mul(100).round(2))

tabel_a = ringkas_a(hasil, ambang_per_persentil)
print(tabel_a.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
tabel_a.to_csv(f"result/hasil_a_{DAY_NAME}.csv", index=False)