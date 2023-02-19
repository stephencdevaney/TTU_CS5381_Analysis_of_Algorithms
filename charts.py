import pandas as pd
import matplotlib.pyplot as plt
import os.path
import numpy as np

rdt_vs_tap = False
rdt_vs_nmd = False
rdt_vs_td = False

if rdt_vs_tap:
    # Export Ratio of Dropped Task vs Task Arrival Probability Info
    file_path = "RDT_vs_TAP.csv"
    if(os.path.exists(file_path)):
        df = pd.read_csv(file_path)
        plt.plot(df["TAP"],df["RDT"], marker='x', color='g', label='DRL')
        plt.xticks(np.arange(.1, 1, .2))
        plt.yticks(np.arange(0, 1.2, .2))
        plt.ylabel('Ratio of Dropped Tasks')
        plt.xlabel('Task Arrival Probability')
        plt.legend(loc="lower right")
        plt.savefig("RDT_vs_TAP.png")
    else:
        print(f"File {file_path} not found!")
    
if rdt_vs_nmd:
    # Export Ratio of Dropped Task vs Number of Mobile Devices
    file_path = "RDT_vs_NMD.csv"
    if(os.path.exists(file_path)):
        df = pd.read_csv(file_path)
        plt.plot(df["NMD"],df["RDT"], marker='x', color='g', label='DRL')
        plt.xticks(np.arange(10, 150, 40))
        plt.yticks(np.arange(0, 1.2, .2))
        plt.ylabel('Ratio of Dropped Tasks')
        plt.xlabel('Number of Mobile Devices')
        plt.legend(loc="upper left")
        plt.savefig("RDT_vs_NMD.png")
    else:
        print(f"File {file_path} not found!")
    
if rdt_vs_td:
    # Export Ratio of Dropped Task vs Task Arrival Probability Info
    file_path = "RDT_vs_TD.csv"
    if(os.path.exists(file_path)):
        df = pd.read_csv(file_path)
        plt.plot(df["TD"],df["RDT"], marker='x', color='g', label='DRL')
        plt.xticks(np.arange(0.6, 2.6, .4))
        plt.yticks(np.arange(0, 1.2, .2))
        plt.ylabel('Ratio of Dropped Tasks')
        plt.xlabel('Task Deadline')
        plt.legend(loc="upper right")
        plt.savefig("RDT_vs_TD.png")
    else:
        print(f"File {file_path} not found!")