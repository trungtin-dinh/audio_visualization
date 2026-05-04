## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Pipeline en un coup d'oeil](#2-pipeline-en-un-coup-doeil)
3. [Notation](#3-notation)
4. [Extraction des caractéristiques audio](#4-extraction-des-caracteristiques-audio)
5. [Construction du spectre de Fourier 2D](#5-construction-du-spectre-de-fourier-2d)
6. [Symétrie hermitienne et reconstruction de l'image](#6-symetrie-hermitienne-et-reconstruction-de-limage)
7. [Modes de sortie](#7-modes-de-sortie)
8. [Méthodes de segmentation](#8-methodes-de-segmentation)
9. [Synthèse par sections](#9-synthese-par-sections)
10. [Algorithmes de disposition des sections](#10-algorithmes-de-disposition-des-sections)
11. [Pipeline de post-traitement](#11-pipeline-de-post-traitement)
12. [Référence des paramètres](#12-reference-des-parametres)
13. [Limites](#13-limites)

---

## 1. Vue d'ensemble

Cette application convertit un signal audio mono en une image carrée en utilisant uniquement des opérations classiques de traitement du signal. Aucun modèle entraîné n'intervient à quelque étape que ce soit. Le même audio, avec les mêmes paramètres, produit toujours la même image.

Le principe central est la **synthèse spectrale inverse en deux dimensions**. Toute image réelle de taille $N \times N$ possède une transformée de Fourier discrète 2D (DFT) unique, c'est-à-dire une matrice complexe dont les coefficients encodent l'amplitude et la phase de sinusoïdes spatiales. La direction inverse, consistant à construire un spectre complexe puis à l'inverser, est tout aussi valide et produit une image spatiale légitime. L'application suit cette voie : elle extrait un ensemble riche de caractéristiques audio, les projette sur le module et la phase d'une matrice complexe synthétique $N \times N$, puis reconstruit l'image par DFT inverse 2D.

Le module d'un coefficient spectral à la fréquence spatiale $(u,v)$ détermine la quantité d'énergie que ce motif sinusoïdal apporte à l'image. La phase détermine où ce motif est positionné dans l'espace. Ces deux degrés de liberté sont remplis à partir de groupes distincts de caractéristiques audio : les représentations fondées sur l'énergie (module de la STFT, module de la CWT, spectrogramme mel, chroma, MFCC, RMS) alimentent la grille de module ; les représentations de phase et les représentations temporelles (phase de la STFT, phase instantanée de la CWT, onset strength, centroid spectral, ZCR) alimentent la grille de phase. L'image obtenue encode la structure spectro-temporelle de l'audio sous une forme visuellement interprétable et entièrement traçable jusqu'à sa source.

$$
x[n]
\;\xrightarrow{\;\text{extraction des caractéristiques}\;}
\bigl\{\mathbf{F}_i\bigr\}
\;\xrightarrow{\;\text{construction des grilles}\;}
\bigl(\widetilde{M}[u,v],\;\widetilde{\Phi}[u,v]\bigr)
\;\xrightarrow{\;\text{IFFT2}\;}
f[x,y]
\;\xrightarrow{\;\text{post-traitement}\;}
\text{image RGB}
$$

---

## 2. Pipeline en un coup d'oeil

Le tableau ci-dessous résume chaque étape du pipeline, ce qu'elle produit, et où cette sortie est utilisée.

| Étape | Sortie | Utilisée par |
|---|---|---|
| STFT multi-résolution | matrices de log-module ; matrices de phase déroulée | Grille de module (module) ; Grille de phase (phase) |
| Transformée en ondelettes continue | module CWT ; phase instantanée CWT (Morlet seulement) | Grille de module ; Grille de phase |
| Spectrogramme mel | matrice d'énergie log-mel | Grille de module |
| Chroma | matrice d'énergie par classe de hauteur | Grille de module |
| MFCC | matrice de coefficients cepstraux | Grille de module |
| Énergie RMS | série temporelle scalaire | Grille de module |
| Centroid spectral | série temporelle scalaire | Grille de phase |
| Onset strength | série temporelle scalaire | Grille de phase |
| Taux de passage par zéro | série temporelle scalaire | Grille de phase |
| Grille de module $\widetilde{M}[u,v]$ | flottant $N \times N$ dans $[0,1]$ | Assemblage du spectre complexe |
| Grille de phase $\widetilde{\Phi}[u,v]$ | flottant $N \times N$ dans $(-\pi, \pi]$ | Assemblage du spectre complexe |
| Symétrisation hermitienne | $Z_\text{sym}$ complexe $N \times N$ | IFFT 2D |
| IFFT2 + normalisation | image spatiale en virgule flottante | Modes de sortie, assemblage par sections |
| Mode de sortie | patch d'image RGB | Canevas sectionné ou sortie finale |
| Post-traitement | image RGB finale $N \times N$ en uint8 | Export |

---

## 3. Notation

| Symbole | Signification |
|---|---|
| $x[n]$ | forme d'onde audio mono, $n = 0, \ldots, L-1$ |
| $L$ | nombre total d'échantillons |
| $f_s$ | fréquence d'échantillonnage en Hz |
| $N$ | longueur du côté de l'image de sortie, en pixels |
| $N_k$ | longueur de fenêtre STFT (puissance de deux) |
| $H_k = N_k/4$ | pas de la STFT (recouvrement de 75 %) |
| $X_{N_k}[m,t]$ | coefficient STFT complexe, bin fréquentiel $m$, trame $t$ |
| $W[s,n]$ | coefficient CWT complexe, échelle $s$, temps $n$ |
| $S$ | nombre d'échelles CWT |
| $B$ | nombre de bandes du banc de filtres mel |
| $C$ | nombre de coefficients MFCC |
| $\widetilde{M}[u,v]$ | grille de module construite, $N \times N$, valeurs dans $[0,1]$ |
| $\widetilde{\Phi}[u,v]$ | grille de phase construite, $N \times N$, valeurs dans $(-\pi,\pi]$ |
| $Z_\text{sym}[u,v]$ | spectre symétrisé hermitien |
| $f[x,y]$ | image spatiale reconstruite, $\operatorname{Re}(\text{IFFT2}(Z_\text{sym}))$ |
| $k$ | nombre de sections temporelles |
| $\overline{\mathbf{A}}$ | tableau $\mathbf{A}$ remis linéairement à l'échelle dans $[0,1]$ |

Le bin fréquentiel $m$ d'une STFT à la fréquence d'échantillonnage $f_s$ et avec une fenêtre $N_k$ correspond à $f_m = m f_s / N_k$ Hz. Le coefficient DC de la DFT 2D se trouve à l'indice $(0,0)$ selon la convention NumPy. Indices spatiaux : $x$ est la colonne (horizontale), $y$ est la ligne (verticale).

---

## 4. Extraction des caractéristiques audio

La forme d'onde est chargée à sa fréquence d'échantillonnage originale sans rééchantillonnage, convertie en mono par moyenne des canaux si nécessaire, puis tronquée à 60 secondes. librosa normalise le signal dans $[-1, 1]$. Toutes les caractéristiques sont extraites directement à $f_s$, ce qui préserve l'interprétation physique exacte de chaque bin fréquentiel.

### 4.1 STFT multi-résolution

La STFT décompose le signal en information temps-fréquence simultanée en appliquant une DFT à des segments fenêtrés recouvrants. Avec une fenêtre de Hann $w[n] = \frac{1}{2}(1 - \cos(2\pi n / N_k))$ et un pas $H_k = N_k / 4$ :

$$
X_{N_k}[m,t] = \sum_{n=0}^{N_k-1} x[n + t H_k]\\, w[n]\\, e^{-j2\pi mn/N_k}, \qquad m = 0, \ldots, \lfloor N_k/2 \rfloor
$$

La fenêtre de Hann réduit la fuite spectrale : sans pondération progressive, la coupure abrupte aux frontières des trames introduit un contenu artificiel en hautes fréquences. Ses lobes secondaires décroissent comme $1/f^3$, ce qui en fait un choix standard pour l'analyse audio.

Une seule longueur de fenêtre impose un compromis strict entre résolution temporelle et résolution fréquentielle, formalisé par le principe d'incertitude de Gabor-Heisenberg $\sigma_t \cdot \sigma_f \geq 1/(4\pi)$. Pour atténuer cela, le pipeline calcule des STFT pour toutes les puissances de deux $N_k \in [N_{\min}, N_{\max}]$ telles que $N_k \leq L/2$, avec par défaut l'ensemble $\{256, 512, 1024, 2048, 4096, 8192\}$. Les STFT à fenêtres courtes ($N_k = 256, 512$) résolvent les transitoires rapides et les attaques ; les STFT à fenêtres longues ($N_k = 4096, 8192$) résolvent les partiels harmoniques proches. Toutes les résolutions contribuent de manière égale à la grille de module finale, avec des poids individuels $w_{N_k} = w_{\text{STFT}} / |\mathcal{R}|$.

Avant le mélange, le module brut est compressé sous la forme $\hat{M} = \log(1 + |X_{N_k}[m,t]|)$. Cette opération approxime l'échelle en décibels et empêche quelques transitoires très forts de dominer toute la dynamique.

La STFT est aussi la source principale de la **grille de phase**. Deux résolutions contribuent directement : $N_k = 1024$ (poids 0.30) comme source de phase principale avec un compromis temps-fréquence équilibré, et $N_k = 512$ (poids 0.20) pour un suivi temporel plus fin de la phase des transitoires rapides. Avant interpolation, la phase enroulée $\angle X_{N_k} \in (-\pi, \pi]$ est déroulée suivant les axes fréquentiel et temporel afin de supprimer les sauts artificiels de $\pm 2\pi$, ce qui produit un champ de phase lisse qui s'interpole vers $N \times N$ sans artefacts de discontinuité.

### 4.2 Transformée en ondelettes continue

La STFT analyse toutes les fréquences avec la même résolution temporelle. La CWT met au contraire à l'échelle la fenêtre d'analyse proportionnellement à la période d'oscillation, donnant une résolution temporelle plus fine aux hautes fréquences et une résolution fréquentielle plus fine aux basses fréquences : c'est la propriété de $Q$ constant, qui se rapproche du comportement de la cochlée. La transformée à l'échelle $s$ et au temps $\tau$ est :

$$
W[s, \tau] = \frac{1}{\sqrt{s}} \sum_{n=0}^{L-1} x[n]\\, \overline{\psi\!\left(\frac{n - \tau}{s}\right)}
$$

Le facteur $1/\sqrt{s}$ normalise l'énergie entre les échelles, et le renversement temporel conjugué transforme l'opération en filtre adapté : $W[s,\tau]$ est grand lorsque le signal ressemble localement à l'ondelette d'échelle $s$ autour de $\tau$.

**Ondelette de Morlet.** Le choix par défaut est $\psi(t) = \pi^{-1/4} e^{j\omega_0 t} e^{-t^2/2}$ avec $\omega_0 = 6$. L'enveloppe gaussienne localise l'ondelette dans le temps ; l'exponentielle complexe la fait osciller à $\omega_0 / s$ rad/échantillon. Point crucial, l'ondelette de Morlet est **analytique** : sa transformée de Fourier est négligeable aux fréquences négatives (l'erreur d'admissibilité en DC vaut $e^{-\omega_0^2/2} = e^{-18} \approx 10^{-8}$). L'analyticité signifie que les coefficients complexes $W[s,\tau]$ possèdent une **phase instantanée** $\angle W[s,\tau]$ bien définie : pour une sinusoïde pure de fréquence $\omega$, cette phase avance exactement au taux $\omega$ selon $\tau$, encodant la vraie phase locale du signal à l'échelle $s$. Cette phase instantanée alimente la grille de phase avec un poids de 0.20.

**Ondelette de Ricker (Mexican hat).** L'alternative est $\psi(t) \propto (1 - t^2/\sigma^2) e^{-t^2/(2\sigma^2)}$, la dérivée seconde d'une gaussienne. Elle est réelle, donc aucun signal analytique n'existe et aucune phase instantanée n'est définie. En mode Ricker, seul $|W[s,\tau]|$ est utilisé ; le poids de phase CWT (0.20) est redistribué vers les sources de phase STFT.

Les échelles sont espacées logarithmiquement de $s = 1$ à $s_{\max} = \min(512, L'/2)$ sur $S = 64$ pas. L'espacement logarithmique donne le même nombre d'échelles par octave. Comme le coût de la CWT croît comme $O(L \cdot S \cdot s_{\max})$, le signal est décimé à au plus $L' = 44\\,100$ échantillons avant la CWT, ce qui préserve la forme spectrale globale tout en gardant le calcul tractable.

### 4.3 Représentations perceptuelles

**Spectrogramme mel.** L'échelle mel est une échelle perceptuelle de hauteur qui modélise la résolution fréquentielle cochléaire : approximativement linéaire sous 1 000 Hz et logarithmique au-dessus. Un banc de $B = 128$ filtres triangulaires uniformément espacés sur l'axe mel projette le spectre de puissance de la STFT vers des énergies par bandes mel :

$$
M[b, t] = \sum_{m=0}^{N_k/2} H_b[m] \cdot |X_{N_k}[m,t]|^2
$$

où $H_b[m]$ est la bande passante triangulaire du filtre $b$. Le résultat compressé logarithmiquement $\log(1 + M[b,t])$ alimente la grille de module avec un poids de 0.18. Par rapport à la STFT brute, le spectrogramme mel met l'accent sur la plage fréquentielle la plus pertinente pour la perception humaine et réduit l'influence des bins supérieurs, où de nombreux bins STFT se projettent vers une seule bande mel.

**Chroma.** Le chroma replie le spectre STFT sur les 12 classes de hauteur de la gamme tempérée (C, C#, …, B) en sommant l'énergie sur toutes les octaves :

$$
C[p, t] = \sum_{\{m\\,:\\,\lfloor 12 \log_2(f_m / f_{\text{ref}})\rfloor \bmod 12\\, =\\, p\}} |X_{N_k}[m,t]|
$$

avec $f_{\text{ref}} = 261.63$ Hz (do central). Le chroma est invariant par octave : un accord de do majeur dans n'importe quel registre produit le même vecteur à 12 dimensions. Il capture le contenu tonal et harmonique indépendamment de la hauteur absolue, avec une contribution à la grille de module de poids 0.09.

**MFCC.** Les Mel-Frequency Cepstral Coefficients sont la DCT-II du spectre log-mel :

$$
\text{cc}[c, t] = \sum_{b=0}^{B-1} \log M[b, t]\;\cos\!\left(\frac{\pi c}{B}\!\left(b + \tfrac{1}{2}\right)\right), \qquad c = 0, \ldots, C-1
$$

avec $C = 20$ coefficients. La DCT-II diagonalise approximativement la matrice de covariance des spectres mel, si bien que les coefficients sont presque décorrélés : chacun transporte une information presque indépendante sur l'enveloppe spectrale. Le coefficient 0 suit la log-énergie ; les coefficients 1 à 13 encodent la forme timbrale globale ; les coefficients plus élevés ajoutent une texture plus fine. Les valeurs absolues $|\text{cc}[c,t]|$ contribuent à la grille de module avec un poids de 0.09.

### 4.4 Caractéristiques scalaires temporelles

Ces caractéristiques produisent une seule série temporelle plutôt qu'une matrice temps-fréquence, et leur rôle principal se situe dans la **grille de phase**, où elles introduisent une structure temporelle sous forme de décalages de phase spatialement variables.

**Énergie RMS** $\text{rms}[t] = \sqrt{\frac{1}{N_{\min}}\sum_n x[n+tH]^2}$ suit la sonie instantanée. Elle contribue avec un faible poids (0.04) à la grille de module comme modulation d'amplitude spatiale : les passages forts produisent davantage d'énergie dans le spectre.

**Taux de passage par zéro** $\text{zcr}[t]$ compte les changements de signe par échantillon dans chaque trame. Un ZCR élevé indique un contenu bruité ou fricatif ; un ZCR faible indique un contenu tonal. Il contribue à la grille de phase avec un poids de 0.05, mis à l'échelle sur une plage de phase de $\pm\pi/4$.

**Centroid spectral** $\mu_f[t] = \sum_m f_m |X|^2 / \sum_m |X|^2$ est la fréquence moyenne pondérée par la puissance, fortement corrélée à la brillance perçue. Il contribue à la grille de phase avec un poids de 0.10, mis à l'échelle sur $\pm\pi$.

**Onset strength** détecte les attaques de notes et les événements rythmiques comme la moyenne de la demi-onde positive rectifiée de la différence d'ordre un du spectrogramme log-mel entre les bandes : $\text{onset}[t] = \frac{1}{B}\sum_b \max(0,\\, \log M[b,t] - \log M[b,t-1])$. Elle est élevée lorsque l'énergie spectrale augmente brusquement, et contribue à la grille de phase avec un poids de 0.15, mis à l'échelle sur $\pm\pi/2$.

Toutes les caractéristiques scalaires suivent le même chemin vers les grilles 2D : normalisation dans $[0,1]$, rééchantillonnage à $N$ points, puis réplication sur les $N$ lignes afin de produire un tableau $N \times N$ dont les colonnes représentent le temps. Cela encode le temps comme axe horizontal de l'image spatiale.

---

## 5. Construction du spectre de Fourier 2D

### 5.1 Interpolation vers $N \times N$

Chaque matrice de caractéristiques a une forme déterminée par ses propres paramètres (taille de fenêtre, pas, nombre d'échelles, etc.) et par la longueur du signal, généralement différente de $N \times N$. Le rééchantillonnage vers $N \times N$ utilise une spline bicubique 2D de degré 3, évaluée sur une grille de destination normalisée $[0,1]^2$. Les splines bicubiques produisent des résultats $C^1$-continus et évitent le ringing de l'interpolation sinc. Si un axe source possède moins de 4 points, le degré de spline est réduit à $\min(3, \text{taille}-1)$.

### 5.2 Découpage en bandes pour le mode Colors

En mode Colors (et dans les modes qui en dérivent), l'axe fréquentiel normalisé de chaque caractéristique est partitionné en trois bandes avant interpolation : Low $[0, \alpha)$, Mid $[\alpha, \beta)$, High $[\beta, 1)$ avec les coupures par défaut $\alpha = 1/3$, $\beta = 2/3$. Chaque bande est interpolée et traitée indépendamment, générant trois canaux spatiaux distincts empilés comme R (Low), G (Mid), B (High). Le résultat est une coloration spectrale de l'image : l'énergie grave contrôle le canal rouge, le médium contrôle le vert, et les hautes fréquences contrôlent le bleu.

### 5.3 Grille de module

La grille de module $\widetilde{M} \in [0,1]^{N \times N}$ est la somme pondérée des six contributions listées ci-dessous. Chaque matrice de caractéristiques est normalisée indépendamment dans $[0,1]$ et interpolée vers $N \times N$ avant le mélange. Tous les poids sont automatiquement normalisés pour sommer à 1.

$$
\widetilde{M} = w_{\text{STFT}}\,\overline{M}_{\text{STFT}} + w_{\text{CWT}}\,\overline{M}_{\text{CWT}} + w_{\text{mel}}\,\overline{M}_{\text{mel}} + w_{\text{chr}}\,\overline{C} + w_{\text{mfcc}}\,\overline{|\text{cc}|} + w_{\text{RMS}}\,\overline{E}
$$

| Contribution | Poids par défaut | Ce que cela encode |
|---|---|---|
| STFT (toutes les résolutions moyennées) | 0.45 | Énergie spectrale multi-échelle |
| Spectrogramme mel | 0.18 | Énergie spectrale pondérée perceptuellement |
| Module CWT | 0.15 | Énergie spectrale à $Q$ constant |
| Chroma | 0.09 | Contenu tonal/harmonique par classe de hauteur |
| MFCC (valeurs absolues) | 0.09 | Forme de l'enveloppe spectrale |
| Énergie RMS | 0.04 | Dynamique de sonie |

### 5.4 Grille de phase

La phase est le facteur dominant dans la structure visuelle d'une image : deux images ayant des spectres de module identiques mais des phases mélangées sont perceptuellement sans rapport. Une démonstration classique montre que, lorsque les spectres de phase de deux images sont échangés, la sortie ressemble à l'image dont la phase a été utilisée, et non à celle dont le module a été utilisé. C'est pourquoi la grille de phase reçoit une attention de conception plus importante que la grille de module.

La grille de phase $\widetilde{\Phi} \in (-\pi, \pi]^{N \times N}$ mélange six sources :

$$
\widetilde{\Phi} = w_{\text{mid}}\,\Phi_{1024} + w_{\text{fine}}\,\Phi_{512} + w_{\text{cwt}}\,\Phi_{\text{CWT}} + w_{\text{onset}}\,\Phi_{\text{onset}} + w_{\text{cen}}\,\Phi_{\text{centroid}} + w_{\text{zcr}}\,\Phi_{\text{ZCR}}
$$

Les phases STFT à $N_k = 1024$ et $N_k = 512$ sont déroulées suivant les deux axes avant interpolation. Le déroulement de phase supprime les sauts artificiels de $\pm 2\pi$ introduits par $\operatorname{atan2}$ : les trames consécutives d'une sinusoïde stationnaire doivent montrer une phase qui avance régulièrement, et non des enroulements aléatoires. La correction cumulée à chaque pas est $2\pi \cdot \operatorname{round}\!\bigl((\phi[m,t] - \phi[m,t-1])/(2\pi)\bigr)$. Le champ déroulé est lisse et s'interpole vers $N \times N$ sans artefacts de discontinuité.

Les caractéristiques scalaires temporelles (onset, centroid, ZCR) sont diffusées vers $N \times N$ (Section 4.4) et mises à l'échelle dans des sous-plages fixes : onset vers $\pm\pi/2$, centroid vers $\pm\pi$, ZCR vers $\pm\pi/4$. Ces plages sont calibrées de sorte que chaque caractéristique module la phase de manière perceptible sans la saturer. Après sommation, le résultat est réenroulé dans $(-\pi, \pi]$.

Poids de phase par défaut (Morlet) : $w_{\text{mid}} = 0.30$, $w_{\text{fine}} = 0.20$, $w_{\text{cwt}} = 0.20$, $w_{\text{onset}} = 0.15$, $w_{\text{cen}} = 0.10$, $w_{\text{zcr}} = 0.05$. En mode Ricker, le poids CWT est redistribué proportionnellement à $w_{\text{mid}}$ et $w_{\text{fine}}$.

---

## 6. Symétrie hermitienne et reconstruction de l'image

L'IFFT 2D d'une matrice complexe arbitraire produit une image spatiale complexe. Pour que la sortie soit réelle, le spectre d'entrée doit satisfaire la condition de **symétrie hermitienne** :

$$
F[u,v] = \overline{F[(-u)\bmod N,\;(-v)\bmod N]} \qquad \forall\, u, v
$$

Le spectre assemblé $Z = \widetilde{M} \cdot e^{j\widetilde{\Phi}}$ ne satisfait généralement pas cette propriété, puisque $\widetilde{M}$ et $\widetilde{\Phi}$ sont construits à partir de caractéristiques audio sans lien structurel entre $(u,v)$ et son indice conjugué hermitien. La matrice hermitienne symétrique la plus proche de $Z$ au sens de la norme de Frobenius est :

$$
Z_\text{sym}[u,v] = \frac{Z[u,v] + \overline{Z[(-u)\bmod N,\;(-v)\bmod N]}}{2}
$$

Cette projection est implémentée comme $Z_\text{sym} = (Z + \overline{Z_r})/2$, où $Z_r = \operatorname{roll}_{+1,+1}(Z[::-1,::-1])$, l'opération flip-then-roll réalisant efficacement la correspondance d'indice conjugué modulaire. À l'indice DC $(0,0)$, $Z_r[0,0] = Z[0,0]$, donc $Z_\text{sym}[0,0] = \operatorname{Re}(Z[0,0]) \in \mathbb{R}$, ce qui garantit que la moyenne spatiale de la sortie est réelle. Après symétrisation, $\bigl|\operatorname{Im}(\text{IFFT2}(Z_\text{sym}))\bigr| \sim 10^{-14}$ (uniquement la précision flottante) et cette partie imaginaire est rejetée.

L'image reconstruite est :

$$
f[x, y] = \operatorname{Re}\!\left(\frac{1}{N^2} \sum_{u=0}^{N-1}\sum_{v=0}^{N-1} Z_\text{sym}[u,v]\; e^{j2\pi(ux+vy)/N}\right)
$$

La partie réelle est une image $N \times N$ en virgule flottante encodant le motif d'interférence de toutes les composantes de fréquences spatiales spécifiées par $Z_\text{sym}$.

---

## 7. Modes de sortie

Cinq modes sont disponibles, organisés de manière hiérarchique. Grayscale et Colors effectuent chacun une reconstruction indépendante. Black mix, Luma mix et Watershed exécutent chacun le pipeline deux fois, une fois en mode Colors et une fois en mode Grayscale, puis combinent les résultats.

**Grayscale.** Une seule paire $(\widetilde{M}, \widetilde{\Phi})$ est construite sur toute la bande. L'IFFT2 produit un canal réel, répliqué sur R, G et B.

**Colors.** L'axe fréquentiel est divisé en trois bandes (Section 5.2). Trois reconstructions IFFT2 indépendantes produisent trois canaux spatiaux empilés comme R, G et B, créant une coloration spectrale où le contenu fréquentiel est directement visible sous forme de couleur.

**Black mix.** Les images Colors et Grayscale sont toutes deux calculées. Le canal en niveaux de gris est optionnellement lissé par gaussienne puis binarisé avec le seuil d'Otsu, qui maximise la variance inter-classe :

$$
\sigma_B^2(\theta) = \omega_0(\theta)\,\omega_1(\theta)\,[\mu_0(\theta) - \mu_1(\theta)]^2
$$

où $\omega_i$ et $\mu_i$ sont respectivement la masse de probabilité et la moyenne de la classe au seuil $\theta$. L'utilisateur choisit quelle classe d'Otsu (sombre, claire, ou la classe minoritaire automatique) forme le masque de dessin ; dans cette classe, les $d\%$ pixels les plus extrêmes sont conservés et optionnellement dilatés morphologiquement de $t$ pixels. Ces pixels sont mis en noir dans l'image Colors.

**Luma mix.** L'image Colors est multipliée canal par canal par une carte de coefficient de luminance dérivée de l'image Grayscale. Le canal en niveaux de gris $g$ est normalisé, optionnellement flouté, élevé à une puissance gamma définie par l'utilisateur $\gamma_\alpha$, puis mélangé avec un plancher minimal $\alpha_{\min}$ à une force $\lambda$ :

$$
I_\text{out}[x,y,c] = I_\text{color}[x,y,c]\cdot\bigl[(1-\lambda) + \lambda(\alpha_{\min} + (1-\alpha_{\min})\cdot g[x,y]^{\gamma_\alpha})\bigr]
$$

Les régions où l'image Grayscale est sombre sont assombries multiplicativement ; les régions claires restent en couleur complète.

**Watershed.** Applique un algorithme de segmentation à l'image Luma mix pour produire une mosaïque de régions remplies. La méthode de segmentation est choisie indépendamment de ce mode de sortie (voir Section 8).

---

## 8. Méthodes de segmentation

La segmentation partitionne l'image Luma mix en régions connexes, chacune remplie avec une couleur représentative. Les cinq méthodes ci-dessous partagent la même stratégie de coloration et la même logique de rendu des frontières : la couleur représentative de chaque région est, par défaut, un **pixel sélectionné aléatoirement** dans la région. Ce choix est délibéré : pour les images de type bruité (ce qui correspond à la sortie typique du pipeline de synthèse spectrale), la moyenne ou la médiane d'une grande région converge vers du gris par la loi des grands nombres, ce qui délave les couleurs. Un pixel aléatoire préserve toute la diversité chromatique de la source. La moyenne et la médiane sont disponibles mais ne sont pas recommandées pour cette application.

Le rendu des frontières est optionnel pour toutes les méthodes : les frontières de régions peuvent être dessinées en noir, comme moyenne de couleur locale, ou laissées invisibles.

### 8.1 Watershed

L'algorithme watershed traite le module du gradient de l'image comme une surface topographique et simule une inondation depuis des marqueurs semences placés dans les vallées du gradient. Le gradient de Sobel $g = \sqrt{g_x^2 + g_y^2}$ est calculé sur l'image en niveaux de gris lissée. Les marqueurs sont initialisés au minimum local du gradient dans chaque cellule $d \times d$ d'une grille régulière. Un flood par tas min étend chaque région étiquetée selon le **coût minimax du chemin** :

$$
\text{cost}(p) = \max\bigl(\text{cost}(\text{parent}(p)),\; g[p]\bigr)
$$

Ce critère arrête l'inondation sur les crêtes du gradient (bords), produisant des frontières de régions alignées sur les bords réels de l'image. Le nombre de régions est contrôlé par l'espacement des marqueurs $d$ : un espacement plus petit produit davantage de régions plus petites.

### 8.2 K-means

Les pixels sont regroupés dans l'espace de couleurs RGB avec l'algorithme K-means standard (`scipy.cluster.vq.kmeans2`). Les $N \times N = n_p$ pixels sont remodelés en un tableau flottant $(n_p, 3)$, blanchis (chaque canal est divisé par son écart-type), puis regroupés en $k$ centroïdes en assignant itérativement chaque pixel au centroïde le plus proche et en recalculant les centroïdes. La valeur par défaut $k = 120$ cible la plage de 100 à 150 régions. Comme les centroïdes sont des points mathématiques dans l'espace de couleurs et non des pixels réels de l'image, ils ne sont pas utilisés pour la coloration : la couleur de région vient de la stratégie du pixel aléatoire.

K-means utilise seulement scipy (déjà une dépendance) et reste toujours disponible comme méthode de secours.

### 8.3 SLIC

SLIC (Simple Linear Iterative Clustering) produit des **superpixels** en regroupant les pixels dans un espace de caractéristiques conjoint $(R, G, B, x, y)$. Les coordonnées spatiales sont incluses avec un poids de compacité qui règle le compromis entre homogénéité de couleur et régularité spatiale : compacité élevée → régions carrées, proches d'une grille ; compacité faible → régions irrégulières, suivant davantage la couleur. SLIC initialise les centres de clusters sur une grille régulière et itère une étape locale de K-means dans une fenêtre de recherche $2s \times 2s$ autour de chaque centre, où $s = \sqrt{n_p / k}$ est l'espacement de grille. Cette localité le rend $O(n_p)$ indépendamment de $k$, contrairement au K-means global. Nécessite scikit-image ; se rabat sur K-means si indisponible.

### 8.4 Felzenszwalb

L'algorithme de Felzenszwalb construit un arbre couvrant de poids minimal du graphe d'adjacence des pixels et fusionne gloutonnement les pixels adjacents. Deux composantes $A$ et $B$ sont fusionnées lorsque le poids d'arête $w(A,B)$ entre elles (la différence minimale de couleur à leur frontière) est petit relativement à la variation interne de chaque composante :

$$
w(A, B) \leq \min\!\left(\max_{e \in A} w(e) + \frac{\tau}{|A|},\;\; \max_{e \in B} w(e) + \frac{\tau}{|B|}\right)
$$

où les termes $\tau/|$région$|$ imposent un minimum de preuve interne avant fusion. Le paramètre d'échelle $\tau$ contrôle directement la granularité des régions : un $\tau$ plus grand produit moins de régions, plus grandes. Le nombre de régions est **déterminé automatiquement** ; il n'est pas nécessaire de spécifier $k$ à l'avance. Nécessite scikit-image ; se rabat sur K-means si indisponible.

### 8.5 Mean-shift

Mean-shift est un algorithme de recherche de modes : chaque échantillon se déplace itérativement vers le centroïde pondéré de son voisinage jusqu'à convergence. Appliqué aux couleurs des pixels, il trouve les modes de la densité de couleurs et regroupe les pixels selon le mode vers lequel ils convergent. La taille du voisinage est contrôlée par la bande passante $h$ (le rayon du noyau dans l'espace de couleur) ; un $h$ plus grand produit moins de régions, plus grandes. Le nombre de clusters est lui aussi automatique.

Comme le coût du mean-shift est $O(n_p^2)$ par itération, l'image est sous-échantillonnée à une longueur de côté maximale définie par l'utilisateur (64 px par défaut) avant clustering. Les labels des pixels originaux sont récupérés en assignant chacun au cluster du pixel sous-échantillonné spatialement le plus proche. La bande passante peut être réglée manuellement ou estimée automatiquement à partir des données avec un quantile de la distribution des distances par paires. Nécessite scikit-learn ; se rabat sur K-means si indisponible.

---

## 9. Synthèse par sections

Par défaut, la forme d'onde complète génère une seule image. Lorsque $k > 1$ sections sont sélectionnées, la forme d'onde est divisée en $k$ segments chronologiques non recouvrants de longueurs presque égales :

$$
x_i[n] = x\!\left[\left\lfloor \tfrac{iL}{k}\right\rfloor + n\right], \quad n = 0,\ldots, L_i - 1, \quad L_i = \left\lfloor \tfrac{(i+1)L}{k}\right\rfloor - \left\lfloor \tfrac{iL}{k}\right\rfloor
$$

Chaque section est traitée indépendamment par l'extraction complète de caractéristiques et le pipeline IFFT2, produisant un patch brut en virgule flottante. Aucune normalisation par patch n'est appliquée : les patches conservent leurs valeurs brutes de sorte que la normalisation globale appliquée après l'assemblage traite tout le canevas uniformément. Cela évite les discontinuités d'intensité aux frontières de patches qui seraient créées par une normalisation indépendante de chaque patch.

Une fois tous les patches placés, une unique normalisation globale par percentiles est appliquée au canevas assemblé :

$$
\hat{f}[x,y,c] = \operatorname{clip}\!\left(\frac{f[x,y,c] - q_{p_1}}{q_{p_2} - q_{p_1}},\; 0,\; 1\right)
$$

avec par défaut $p_1 = 1\%$, $p_2 = 99\%$. Les percentiles, plutôt que le minimum et le maximum globaux, offrent une robustesse face à des pixels extrêmes isolés.

Le nombre maximal autorisé de sections est $k_{\max} = \min(\lfloor L / L_{\min}\rfloor,\; \lfloor N / 32\rfloor^2,\; 64)$, où $L_{\min} = 16\\,384$ échantillons est la longueur minimale pour que toutes les caractéristiques restent significatives (elle doit permettre la plus grande fenêtre STFT, la plage d'échelles CWT et plusieurs trames MFCC).

---

## 10. Algorithmes de disposition des sections

Chaque disposition assigne chaque pixel du canevas $N \times N$ à l'une des $k$ sections. La section 0 correspond toujours au début de l'audio, la section $k-1$ à la fin. Six arrangements spatiaux sont disponibles.

**Treemap chronologique.** Le canevas est récursivement divisé en deux suivant son côté le plus long, la position de coupe étant proportionnelle au nombre de sections assignées à chaque moitié : $W_1 = \lfloor W \lceil k/2 \rceil / k \rfloor$. Cette stratégie slice-and-dice produit un rectangle par section avec des aires presque égales. Pour le treemap, chaque patch est généré à la taille $\max(W_\text{rect}, H_\text{rect})$, puis redimensionné bicubiquement et recadré au centre pour s'adapter à son rectangle.

**Tranches circulaires horaires.** L'angle polaire horaire de chaque pixel depuis la verticale $\theta = \operatorname{atan2}(x-c,\\, c-y) \bmod 2\pi$ détermine sa section : $i = \lfloor k\theta / (2\pi)\rfloor$. Les patches de sections sont générés à la moitié du côté du canevas (réduction du calcul), puis agrandis avant masquage.

**Cercles concentriques.** L'indice de section est $i = \lfloor k \cdot r / r_{\max}\rfloor$, où $r$ est la distance euclidienne au centre. Les rayons implicites $r_i = r_{\max}\sqrt{i/k}$ donnent à chaque anneau une aire égale $\pi r_{\max}^2 / k$.

**Carrés concentriques.** Même principe en utilisant la distance de Chebyshev $r_\infty = \max(|x-c|, |y-c|)$. La section 0 occupe le carré central ; la section $k-1$ le cadre le plus extérieur.

**Bandes verticales / horizontales.** $i = \lfloor k \cdot x / N\rfloor$ (vertical, temps de gauche à droite) ou $i = \lfloor k \cdot y / N\rfloor$ (horizontal, temps de haut en bas).

---

## 11. Pipeline de post-traitement

Une fois le canevas en virgule flottante assemblé et normalisé globalement dans $[0,1]$, six opérations sont appliquées dans l'ordre. Toutes sont réalisées en virgule flottante ; le résultat est tronqué dans $[0,1]$ et converti en `uint8` seulement à la toute fin.

**Balance par canal.** Des gains indépendants $g_R, g_G, g_B \in [0,3]$ modulent les trois canaux avant normalisation, déplaçant l'équilibre chromatique global.

**Normalisation robuste.** La normalisation par percentiles décrite en Section 9 est appliquée ici aussi dans le cas à une seule section.

**Contraste.** Un étirement linéaire centré en 0.5 : $\hat{I} \leftarrow \operatorname{clip}(0.5 + c_s(\hat{I} - 0.5), 0, 1)$. Les valeurs $c_s > 1$ éloignent les intensités du point moyen ; $c_s < 1$ les comprime.

**Luminosité.** Une simple mise à l'échelle multiplicative : $\hat{I} \leftarrow \operatorname{clip}(b \cdot \hat{I},\\, 0,\\, 1)$.

**Correction gamma.** La transformation en loi de puissance $\hat{I} \leftarrow \hat{I}^\gamma$ éclaircit ($\gamma < 1$) ou assombrit ($\gamma > 1$) l'image de manière non linéaire. La valeur par défaut $\gamma = 0.85$ compense partiellement la tendance à la sous-exposition des caractéristiques spectrales compressées logarithmiquement, qui se concentrent près de zéro pour les signaux faibles.

**Mise à l'échelle de la saturation.** La luminance ITU-R BT.601 $Y = 0.299R + 0.587G + 0.114B$ est calculée et utilisée pour interpoler entre niveaux de gris et couleur complète : $I'^{(c)} = \operatorname{clip}(Y + s_\text{sat}(I^{(c)} - Y), 0, 1)$. À $s_\text{sat} = 0$, la sortie est entièrement désaturée ; à $s_\text{sat} = 1$, la saturation est inchangée ; $s_\text{sat} > 1$ sursature.

---

## 12. Référence des paramètres

| Paramètre | Défaut | Plage | Étape du pipeline |
|---|---|---|---|
| Taille d'image $N$ | 512 px | 64-1024, pas 16 | Toutes les grilles |
| Mode de sortie | Watershed | 5 options | §7 |
| Disposition des sections | None | 7 options | §10 |
| Sections $k$ | 32 | 1-$k_{\max}$ | §9 |
| Ondelette CWT | Morlet | Morlet / Ricker | §4.2 |
| Fenêtre STFT min | 256 | {256, 512, …, 8192} | §4.1 |
| Fenêtre STFT max | 8192 | {256, 512, …, 8192} | §4.1 |
| Échelles CWT $S$ | 64 | 16-128 | §4.2 |
| Échantillons max CWT | 44 100 | 4096-220 500 | §4.2 |
| Bandes mel $B$ | 128 | 32-256 | §4.3 |
| Coefficients MFCC $C$ | 20 | 8-64 | §4.3 |
| Poids de module | voir §5.3 | 0-1, auto-normalisés | §5.3 |
| Poids de phase | voir §5.4 | 0-1, auto-normalisés | §5.4 |
| Coupures fréquentielles $\alpha, \beta$ | 1/3, 2/3 | 0.10-0.45, 0.55-0.90 | §5.2 |
| Mode de normalisation RGB | Per-channel | Per-channel / Shared | §11 |
| Balance RGB $g_c$ | 1.0 | 0-3 | §11 |
| Percentiles de normalisation $p_1, p_2$ | 1 %, 99 % | 0-10 %, 90-100 % | §9, §11 |
| Gamma $\gamma$ | 0.85 | 0.20-2.50 | §11 |
| Contraste $c_s$ | 1.0 | 0.20-3.0 | §11 |
| Luminosité $b$ | 1.0 | 0.20-2.5 | §11 |
| Saturation $s_\text{sat}$ | 1.0 | 0-3 | §11 |
| Black mix - classe Otsu | Auto minority | Auto / Dark / Bright | §7 |
| Black mix - densité de pixels | 50 % | 1-100 % | §7 |
| Black mix - lissage $\sigma$ | 0 | 0-10 | §7 |
| Black mix - épaisseur | 0 px | 0-8 | §7 |
| Luma - force $\lambda$ | 1.0 | 0-1 | §7 |
| Luma - coefficient min | 0 | 0-1 | §7 |
| Luma - gamma du coeff. $\gamma_\alpha$ | 1.0 | 0.20-3.0 | §7 |
| Luma - flou du coeff. $\sigma$ | 0 | 0-12 | §7 |
| Méthode de segmentation | Watershed | 5 méthodes | §8 |
| Mode de couleur des régions | Random pixel | Random pixel / Mean / Median | §8 |
| Style de frontière | None | None / Black / Local mean | §8 |
| Épaisseur de frontière | 0 px | 0-8 | §8 |
| Watershed - espacement marqueurs | 36 px | 4-160 | §8.1 |
| Watershed - gradient $\sigma$ | 1.3 | 0-8 | §8.1 |
| K-means - clusters $k$ | 120 | 10-400 | §8.2 |
| SLIC - segments | 120 | 10-400 | §8.3 |
| SLIC - compacité | 10.0 | 0.1-50 | §8.3 |
| Felzenszwalb - échelle $\tau$ | 100 | 1-500 | §8.4 |
| Felzenszwalb - taille min | 20 px | 1-200 | §8.4 |
| Mean-shift - bande passante $h$ | 0 (auto) | 0-100 | §8.5 |
| Mean-shift - côté max | 64 px | 16-128 | §8.5 |

---

## 13. Limites

**La phase domine l'apparence, plus qu'on pourrait s'y attendre.** La grille de module détermine la granularité et la distribution d'énergie de l'image ; la grille de phase détermine sa structure visuelle. En pratique, modifier les poids de module produit des variations texturales subtiles, tandis que modifier les poids de phase produit des changements structurels drastiques. Si la sortie n'est pas satisfaisante, les sources de phase, et en particulier leurs poids individuels, sont les paramètres les plus productifs à explorer.

**Le spectre construit n'est pas la DFT d'une image naturelle.** Le module et la phase sont construits à partir de caractéristiques audio indépendantes, sans contrainte les reliant entre eux ni à un a priori d'image. L'IFFT2 produit toujours une image spatiale valide, mais ses propriétés statistiques sont entièrement déterminées par la combinaison de caractéristiques plutôt que par un modèle génératif d'images naturelles. Ce n'est pas un défaut ; c'est le caractère fondamental de l'approche.

**Les axes spatiaux n'ont pas de signification physique de principe.** Les caractéristiques audio sont asymétriques : l'axe temporel et l'axe fréquentiel ont des unités physiques et des sémantiques différentes. La grille de Fourier 2D est symétrique en $u$ et $v$. Il n'existe pas de projection de principe du temps ou de la fréquence vers une direction spatiale spécifique. Le temps est partiellement encodé dans les trajectoires de phase et, en mode sectionné, dans le pavage spatial ; mais les directions spatiales verticale et horizontale ne portent aucune signification garantie.

**Les frontières de sections reflètent de vrais changements spectraux.** Des patches de sections adjacentes peuvent présenter des différences visibles de luminosité ou de couleur à leurs frontières. Ce n'est pas un artefact du pipeline : c'est une représentation exacte de changements réels du caractère spectral du signal entre les sections. Dans les dispositions où le temps s'écoule spatialement (bandes, treemap), ces discontinuités encodent des transitions significatives : silence vers attaque, changement de timbre, changement de section structurelle.

**Le coût de calcul croît avec les sections et la CWT.** Les termes de coût dominants sont la CWT ($O(L' \cdot S \cdot s_{\max})$, contrôlée par le paramètre de nombre maximal d'échantillons) et le sectionnement ($k$ fois le coût d'un passage unique). Les modes composites (Black mix, Luma mix, Watershed) exécutent le pipeline complet deux fois, ce qui double le temps total. Pour une exploration interactive, $k \leq 8$ et $N \leq 512$ sont recommandés sur un seul coeur CPU.

---
