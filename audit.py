import pandas as pd

MONDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Monday-WorkingHours.pcap_ISCX.csv"
TUESDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"
THURSDAY_MORNING_WEB_ATT = "datasets/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
THURSDAY_AFTERNOON_INFIL = "datasets/GeneratedLabelledFlows/TrafficLabelling/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"
WEDNESDAY = "datasets/GeneratedLabelledFlows/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"
FRIDAY_MORNING = "datasets/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv"
FRIDAY_AFTERNOON_NMAP = "datasets/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"
FRIDAY_AFTERNOON_DDOS = "datasets/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

# BEKAS GABUNGIN FRIDAY
# dfmorn = pd.read_csv(FRIDAY_MORNING)
# dfaftnmap = pd.read_csv(FRIDAY_AFTERNOON_NMAP)
# dfaftddos = pd.read_csv(FRIDAY_AFTERNOON_DDOS)
# df_mix = pd.concat([dfmorn, dfaftnmap, dfaftddos], ignore_index=True)
# df_mix.to_csv("Friday-WorkingHours.pcap_ISCX-Mix.csv", index=False)
# df = pd.read_csv("Friday-WorkingHours.pcap_ISCX-Mix.csv")

df = pd.read_csv(MONDAY)
print(df.info())