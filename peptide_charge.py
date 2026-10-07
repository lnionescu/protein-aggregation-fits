import numpy as np
import matplotlib.pyplot as plt

seq = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"

# pKa values: CRC Handbook of Chemistry and Physics, 87th edition
# (as used by Innovagen's Peptide Property Calculator)
pKa_acidic = {'D': 3.65, 'E': 4.25, 'C': 8.3, 'Y': 10.07}
pKa_basic  = {'H': 6.0, 'K': 10.53, 'R': 12.48}
pKa_Nterm, pKa_Cterm = 8.0, 3.3

def net_charge(pH, seq):
    Z = 1 / (1 + 10**(pH - pKa_Nterm))       # N-terminus, basic
    Z -= 1 / (1 + 10**(pKa_Cterm - pH))      # C-terminus, acidic
    # formula as given by pepcalc
    for aa in seq:
        if aa in pKa_acidic:
            Z -= 1 / (1 + 10**(pKa_acidic[aa] - pH))
        if aa in pKa_basic:
            Z += 1 / (1 + 10**(pH - pKa_basic[aa]))
    return Z

pH_fine = np.linspace(4, 10, 500)
Z_fine = net_charge(pH_fine, seq)

pI = 5.17  # as reported by PepCalc (Innovagen Peptide Property Calculator)

exp_pH = [6, 6.5, 7, 7.5, 8, 8.5]
exp_colours = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#313695']
exp_Z = [net_charge(p, seq) for p in exp_pH]

fig, ax = plt.subplots(figsize=(5.5, 3.8))

ax.axvspan(pI, max(exp_pH), color='0.93', zorder=0)
ax.plot(pH_fine, Z_fine, color='k', lw=2, zorder=2)
ax.axhline(0, color='0.5', lw=0.8, ls='--', zorder=1)

# mark pI
ax.plot(pI, 0, 'o', color='grey', ms=8, zorder=3,
         markeredgecolor='k', markeredgewidth=0.6,
         label=f'pI = {pI:.2f}')

# show pHs in our data and add extremes to legend
for i, (p, z, c) in enumerate(zip(exp_pH, exp_Z, exp_colours)):
    lbl = f'pH {p}, $Z$ = {z:.1f}' if i in (0, len(exp_pH)-1) else None
    ax.plot(p, z, 'o', color=c, ms=7, zorder=4,
             markeredgecolor='k', markeredgewidth=0.5, label=lbl)

ax.set_xlabel('pH')
ax.set_ylabel('Net charge, $Z$')
ax.set_xlim(4.5, 9.5)
ax.set_title('Aβ42 net charge vs. pH')
ax.legend(loc='upper right', fontsize=8.5, frameon=True, framealpha=0.9)
plt.tight_layout()
plt.show()
