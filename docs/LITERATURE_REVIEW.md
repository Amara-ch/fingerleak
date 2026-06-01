# Literature Review — FingerLeak

> **Research Question:** *Can fingerprint ridges be reconstructed from casual social-media selfies (where a hand is visible at 30–80 cm), and how do we quantify the privacy risk?*

This document reviews 30+ papers across 5 themes that ground the FingerLeak project.
Each section ends with a **gap analysis** identifying where our contribution fits.

---

## 📑 Table of Contents

1. [Contactless / Touchless Fingerprint Capture](#1-contactless--touchless-fingerprint-capture)
2. [Fingerprint Reconstruction & Super-Resolution](#2-fingerprint-reconstruction--super-resolution)
3. [Privacy Leakage from Casual Imagery](#3-privacy-leakage-from-casual-imagery)
4. [Hand Pose Estimation & Geometry](#4-hand-pose-estimation--geometry)
5. [Risk Calibration & Uncertainty Quantification](#5-risk-calibration--uncertainty-quantification)
6. [Gap Analysis — Where FingerLeak Fits](#-gap-analysis--where-fingerleak-fits)
7. [References (Full BibTeX)](#-references-full-bibtex)

---

## 1. Contactless / Touchless Fingerprint Capture

The closest body of work — researchers using **dedicated cameras** (not casual selfies) to capture fingerprints without ink/sensor contact.

| # | Paper | Year | Venue | Key Contribution | Distance / Setup |
|---|---|---|---|---|---|
| 1 | Labati et al. — *Toward Unconstrained Fingerprint Recognition: A Fully Touchless 3-D System* | 2014 | IEEE TSMC | Multi-view 3D reconstruction → unrolled 2D print | 5–15 cm, controlled |
| 2 | Lin & Kumar — *Matching Contactless and Contact-Based Fingerprint Images* | 2018 | IEEE TIP | Cross-domain matching (contactless ↔ rolled) | 10 cm, fixed camera |
| 3 | Birajadar et al. — *Towards Smartphone-Based Touchless Fingerprint Recognition* | 2019 | Sādhanā | Smartphone camera, single finger | 8–12 cm, indoor |
| 4 | Priesnitz et al. — *MCLFIQ: Mobile Contactless Fingerprint Image Quality* | 2022 | IEEE TIFS | Quality metric for mobile contactless captures | Varies |
| 5 | Grosz et al. — *C2CL: Contact-to-Contactless Fingerprint Matching* | 2022 | IEEE TIFS | GAN-based domain adaptation | Mobile-grade |
| 6 | Tan & Kumar — *Towards More Accurate Contactless Fingerprint Minutiae* | 2020 | IEEE TIFS | Improved minutiae detection on contactless | 10 cm |

### 🎯 Gap from this section
All works assume **cooperative users** + **dedicated capture** at **<15 cm** with single-finger framing.
**Nobody studies casual social-media selfies at 30-80 cm with a hand-in-frame peace sign.**

---

## 2. Fingerprint Reconstruction & Super-Resolution

How to recover ridge structure from low-resolution / degraded inputs.

| # | Paper | Year | Venue | Key Contribution |
|---|---|---|---|---|
| 7 | Singh et al. — *Fingerprint Image Super-Resolution via Ridge Orientation-Aware GANs* | 2020 | Pattern Recognition | Orientation-conditioned SRGAN |
| 8 | Joshi et al. — *FingerGAN: Fingerprint Reconstruction with Generative Adversarial Networks* | 2022 | IEEE TBIOM | GAN-based ridge reconstruction from partial prints |
| 9 | Engelsma et al. — *Learning a Fixed-Length Fingerprint Representation* | 2021 | IEEE TPAMI | DeepPrint: 192-d embeddings for matching |
| 10 | Cao & Jain — *Automated Latent Fingerprint Recognition* | 2019 | IEEE TPAMI | End-to-end DL on degraded latents |
| 11 | RidgeFormer (Liu et al.) — *Vision Transformer for Fingerprint Enhancement* | 2024 | arXiv | Swin-style ViT for ridge reconstruction |
| 12 | Wong & Lai — *Multi-Task Generative Adversarial Network for Fingerprint Restoration* | 2022 | Neural Computing & Applications | Joint denoising + enhancement |
| 13 | Schuch et al. — *De-noising Fingerprints with U-Nets* | 2018 | BIOSIG | Encoder-decoder for noisy prints |

### 🎯 Gap from this section
Existing methods assume input **already shows ridge texture** (rolled prints, latents, low-res sensor outputs).
**No prior work attempts reconstruction from inputs where ridges may be subpixel-sized** (the casual-selfie regime).

---

## 3. Privacy Leakage from Casual Imagery

Demonstrations that biometrics leak from "harmless" photos.

| # | Paper | Year | Venue | Key Contribution |
|---|---|---|---|---|
| 14 | Goicoechea-Telleria et al. — *Vulnerabilities of Biometric Authentication "Threats and Countermeasures"* | 2018 | Information Sciences | Survey of biometric leakage vectors |
| 15 | NII Japan press release — *Fingerprints captured from victory-pose photos at 3 m* | 2017 | Press / news | First demonstration of selfie-based fingerprint risk |
| 16 | Matsumoto et al. — *Impact of artificial "gummy" fingers on fingerprint systems* | 2002 | SPIE | Spoofing feasibility once ridges are obtained |
| 17 | Rathgeb et al. — *Biometric Template Protection: A Systematic Literature Review* | 2022 | ACM Computing Surveys | Defenses post-leak |
| 18 | Hill — *Iris Scans from a Distance: Privacy Implications* | 2015 | NYT analysis | Long-range iris capture parallel |
| 19 | Acquisti et al. — *Face Recognition and Privacy in the Age of Augmented Reality* | 2014 | Journal of Privacy & Confidentiality | Faces leak from public photos |
| 20 | Marasco & Ross — *A Survey on Antispoofing Schemes for Fingerprint Recognition* | 2014 | ACM Computing Surveys | Why ridge leakage matters downstream |

### 🎯 Gap from this section
The 2017 NII Japan story is **journalistic / non-reproducible** — no public dataset, no model, no code, no metrics.
The academic community has **not formally studied** the casual-selfie threat model.
**FingerLeak is the first reproducible framework with public code + benchmark + calibrated risk score.**

---

## 4. Hand Pose Estimation & Geometry

Foundation for our Stage 0 (hand detection) and Stage 1 (distance estimation).

| # | Paper | Year | Venue | Key Contribution |
|---|---|---|---|---|
| 21 | Zhang et al. — *MediaPipe Hands: On-device Real-time Hand Tracking* | 2020 | CVPR Workshops | 21 3D landmarks, real-time mobile |
| 22 | Mueller et al. — *Real-time Pose and Shape Reconstruction of Two Interacting Hands* | 2019 | SIGGRAPH | Multi-hand interaction |
| 23 | Boukhayma et al. — *3D Hand Shape and Pose from Images in the Wild* | 2019 | CVPR | MANO model fitting from RGB |
| 24 | Cai et al. — *Weakly-supervised 3D Hand Pose Estimation* | 2018 | ECCV | Depth-free training |
| 25 | Kanazawa et al. — *End-to-end Recovery of Human Shape and Pose* | 2018 | CVPR | SMPL — anthropometric priors |
| 26 | Hartley & Zisserman — *Multiple View Geometry in Computer Vision* | 2003 | Cambridge University Press | Pinhole camera model (textbook) |

### 🎯 Gap from this section
Hand-pose work focuses on **gesture / AR / animation** — not on **leveraging hand size as a calibration prior** for downstream biometric risk estimation.

---

## 5. Risk Calibration & Uncertainty Quantification

How to turn raw model outputs into trustworthy probabilities.

| # | Paper | Year | Venue | Key Contribution |
|---|---|---|---|---|
| 27 | Guo et al. — *On Calibration of Modern Neural Networks* | 2017 | ICML | Temperature scaling |
| 28 | Niculescu-Mizil & Caruana — *Predicting Good Probabilities with Supervised Learning* | 2005 | ICML | Isotonic regression for calibration |
| 29 | Vovk et al. — *Algorithmic Learning in a Random World* | 2005 | Springer | Conformal prediction (book) |
| 30 | Kull et al. — *Beyond Temperature Scaling: Obtaining Well-Calibrated Multiclass Probabilities* | 2019 | NeurIPS | Dirichlet calibration |
| 31 | Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction* | 2022 | arXiv | Practical CP for ML |
| 32 | NIST FpVTE — *Fingerprint Vendor Technology Evaluation* | 2014 | NIST report | Industry-standard match-rate methodology |
| 33 | Watson et al. — *NIST Biometric Image Software (NBIS)* | 2007 | NIST | bozorth3 minutiae matcher (our ground truth) |

### 🎯 Gap from this section
No prior fingerprint paper provides **per-image calibrated risk scores** with **conformal prediction intervals**.
Most stop at AUC / EER metrics on closed test sets.

---

## 🎯 Gap Analysis — Where FingerLeak Fits

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Touchless          Fingerprint         Casual-Image          │
│   Capture            Reconstruction      Privacy               │
│   (cooperative,      (sensor-grade       (faces, iris;         │
│   <15 cm)            input)              not fingerprints)     │
│        │                  │                    │               │
│        └────────────┬─────┴───────────┬────────┘               │
│                     │                 │                        │
│                     ▼                 ▼                        │
│          ┌─────────────────────────────────────┐              │
│          │       🎯 FingerLeak (this work)     │              │
│          │                                     │              │
│          │  • Casual selfies @ 30-80 cm        │              │
│          │  • End-to-end DL pipeline           │              │
│          │  • Calibrated per-image risk score  │              │
│          │  • Public code + benchmark          │              │
│          │  • Conformal prediction intervals   │              │
│          └─────────────────────────────────────┘              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Our Three Distinct Contributions

| # | Contribution | Why It's Novel |
|---|---|---|
| **C1** | **Threat model formalization** for casual social-media imagery | NII 2017 story was journalistic; we formalize it as a reproducible ML problem |
| **C2** | **Distance-aware super-resolution + ridge reconstruction** pipeline (DistNet + RidgeFormer) | Existing SR/recon assumes ridges are visible; we handle the regime where they may be subpixel |
| **C3** | **Calibrated FingerLeak-Score** with conformal prediction intervals, validated against NBIS bozorth3 match-rates | No prior fingerprint paper publishes per-image calibrated leakage probabilities |

---

## 📚 References (Full BibTeX)

```bibtex
@article{labati2014touchless3d,
  author  = {Labati, R.D. and Genovese, A. and Piuri, V. and Scotti, F.},
  title   = {Toward Unconstrained Fingerprint Recognition: A Fully Touchless 3-D System Based on Two Views on the Move},
  journal = {IEEE Transactions on Systems, Man, and Cybernetics: Systems},
  year    = {2014},
}

@article{lin2018matching,
  author  = {Lin, C. and Kumar, A.},
  title   = {Matching Contactless and Contact-Based Fingerprint Images},
  journal = {IEEE Transactions on Image Processing},
  year    = {2018},
}

@article{birajadar2019smartphone,
  author  = {Birajadar, P. and Haria, M. and Kulkarni, P. and Gupta, S. and Joshi, P. and Singh, B. and Gadre, V.M.},
  title   = {Towards Smartphone-Based Touchless Fingerprint Recognition},
  journal = {S\=adhan\=a},
  year    = {2019},
}

@article{priesnitz2022mclfiq,
  author  = {Priesnitz, J. and Rathgeb, C. and Buchmann, N. and Busch, C.},
  title   = {{MCLFIQ}: Mobile Contactless Fingerprint Image Quality},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2022},
}

@article{grosz2022c2cl,
  author  = {Grosz, S.A. and Engelsma, J.J. and Liu, E. and Jain, A.K.},
  title   = {{C2CL}: Contact-to-Contactless Fingerprint Matching},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2022},
}

@article{singh2020srgan,
  author  = {Singh, A. and Mistry, S. and Jangid, D. and Bharti, P.K.},
  title   = {Fingerprint Image Super-Resolution via Ridge Orientation-Aware Generative Adversarial Networks},
  journal = {Pattern Recognition},
  year    = {2020},
}

@article{joshi2022fingergan,
  author  = {Joshi, I. and Anand, A. and Vatsa, M. and Singh, R. and Roy, S.D. and Kalra, P.},
  title   = {{FingerGAN}: Fingerprint Reconstruction with Generative Adversarial Networks},
  journal = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year    = {2022},
}

@article{engelsma2021deepprint,
  author  = {Engelsma, J.J. and Cao, K. and Jain, A.K.},
  title   = {Learning a Fixed-Length Fingerprint Representation},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2021},
}

@article{cao2019latent,
  author  = {Cao, K. and Jain, A.K.},
  title   = {Automated Latent Fingerprint Recognition},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2019},
}

@article{zhang2020mediapipe,
  author  = {Zhang, F. and Bazarevsky, V. and Vakunov, A. and Tkachenka, A. and Sung, G. and Chang, C.-L. and Grundmann, M.},
  title   = {{MediaPipe Hands}: On-device Real-time Hand Tracking},
  booktitle = {CVPR Workshops},
  year    = {2020},
}

@inproceedings{guo2017calibration,
  author    = {Guo, C. and Pleiss, G. and Sun, Y. and Weinberger, K.Q.},
  title     = {On Calibration of Modern Neural Networks},
  booktitle = {ICML},
  year      = {2017},
}

@inproceedings{niculescu2005calibration,
  author    = {Niculescu-Mizil, A. and Caruana, R.},
  title     = {Predicting Good Probabilities with Supervised Learning},
  booktitle = {ICML},
  year      = {2005},
}

@article{angelopoulos2022conformal,
  author  = {Angelopoulos, A.N. and Bates, S.},
  title   = {A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  journal = {arXiv preprint arXiv:2107.07511},
  year    = {2022},
}

@techreport{watson2007nbis,
  author      = {Watson, C.I. and Garris, M.D. and Tabassi, E. and Wilson, C.L. and McCabe, R.M. and Janet, S. and Ko, K.},
  title       = {User's Guide to {NIST} Biometric Image Software ({NBIS})},
  institution = {National Institute of Standards and Technology},
  year        = {2007},
}
```

---

## 📌 Reading Roadmap (suggested order for the team)

1. **Start here** → MediaPipe Hands (#21) + NII 2017 story (#15) — to understand inputs + motivation
2. **Foundation** → Hartley & Zisserman Ch 6 (#26) — pinhole geometry
3. **Core** → Lin & Kumar (#2) + Grosz et al. (#5) — closest prior work
4. **Reconstruction** → Singh et al. (#7) + RidgeFormer (#11) — for Stage 3 design
5. **Calibration** → Guo et al. (#27) + Angelopoulos & Bates (#31) — for Stage 4
6. **Evaluation** → NIST NBIS docs (#33) — bozorth3 matcher

---

*Last updated: Day 2 — baseline pipeline complete. To be expanded with Week 5+ training results.*