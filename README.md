# fingerleak
Privacy-preserving detection of fingerprint leakage from casual selfies using deep learning
# FingerLeak 🔒✌️

> *"Can a casual selfie leak your fingerprint? We measure it, and we mitigate it."*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: WIP](https://img.shields.io/badge/status-active%20research-green.svg)](#roadmap)

**FingerLeak** is a research project that quantifies and mitigates the privacy risk of latent
fingerprint leakage from casual selfies — particularly the ubiquitous peace-sign (✌️) pose.

## 🧠 Background

In 2017, NII researchers in Tokyo demonstrated that fingerprints could be reconstructed from
photos taken up to 3 metres away. Recent reporting (SCMP 2024–25) and security articles
(Cybernews) have highlighted growing concern about *latent fingerprint leakage* from social
media. Yet, **no public, calibrated metric exists** that tells a user how risky a given selfie
actually is. FingerLeak fills that gap.

## 🎯 Key Contributions (Planned)

1. **C1: FingerLeak-Score** — a calibrated, reproducible privacy-risk metric per selfie.
2. **C2: DistNet** — distance-conditioned super-resolution for fingertip ROIs.
3. **C3: RidgeFormer** — transformer-based ridge reconstruction trained on contactless prints.
4. **C4: Wild evaluation** — first benchmark on selfies "in the wild" with consent-protocol data.

## 🏗️ System Architecture

```mermaid
flowchart LR
    A[📸 Selfie] --> B[Stage 0: MediaPipe Hand + Peace-Sign]
    B --> C[Stage 1: Distance + Fingertip Crops]
    C --> D[Stage 2: DistNet SR<br/><i>planned</i>]
    D --> E[Stage 3: Gabor → RidgeFormer]
    E --> F[Stage 4: FingerLeak-Score]
    F --> G[🛡️ Risk Report<br/>Low / Medium / High / Critical]
