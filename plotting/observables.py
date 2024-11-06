import matplotlib.pyplot as plt
import scienceplots
import numpy as np

plt.style.use(['science', 'high-contrast'])

def plot_observables(unosc: dict, osc: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    x = []
    weights = []
    labels = []

    for flavor in unosc.keys():
        x.append(unosc[flavor]["energy"][0][:-1])
        weights.append(unosc[flavor]["energy"][1])
        labels.append(flavor)
    
    ax[0].hist(x, bins=unosc["nuE"]["energy"][0], weights=weights,
                stacked=True, 
                histtype='step',
                edgecolor='black')
    
    ax[0].hist(x, bins=unosc["nuE"]["energy"][0], weights=weights,
            stacked=True, 
            label=labels,
            alpha=1,
            color=["#bb27f6", "#8f7252", "#f0995c", "#f9dc81", "green", "black", "blue", "red"])
    
    weights = 0
    for flavor in osc.keys():
        if flavor == "nuS" or flavor == "nuSBar":
            continue
        weights += osc[flavor]["energy"][1]
    
    ax[0].plot(osc["nuE"]["energy"][0][:-1], weights, label="Oscillated", color="black", ls="none", marker="o", markersize=1)

    ax[0].legend()

    plt.plot()
    plt.show()

    return