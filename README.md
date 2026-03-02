# Projet Deep Learning : Traduction Multilingue de Vidéos
  
![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c) ![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9D423) ![Poetry](https://img.shields.io/badge/Poetry-Package%20Manager-cyan) ![Status](https://img.shields.io/badge/Status-Complété-success)

Ce dépôt contient l'intégralité du code, des données et de l'analyse réalisés dans le cadre du projet de fin de cours de **Deep Learning (Master)**.
  
L'objectif de ce projet est de concevoir un pipeline de bout en bout capable de transcrire l'audio d'une vidéo francophone (ASR) et de traduire automatiquement les sous-titres générés vers **5 langues** (Anglais, Espagnol, Portugais, Italien, Allemand), tout en évaluant l'impact des erreurs de la chaîne (Cascade Errors).
  
---
## Table des matières
1. [[#Architecture du Pipeline]]
2. [[#Structure du projet]]
3. [[#Installation et Prérequis]]
4. [[#Utilisation]]
5. [[#Résultats Quantitatifs]]
6. [[#Analyse Qualitative (Cascade Errors)]]
  
---
  
## Architecture du Pipeline
  
Le projet est divisé en deux grandes tâches de Deep Learning, reposant sur le dataset **mTEDx (Multilingual TEDx)** :
  
### 1. Automatic Speech Recognition (ASR)
* **Modèle :** `openai/whisper-small` (OpenAI).
* **Fonctionnalité :** Extraction du texte francais à partir de l'audio (`.flac` / `.wav`) et génération de fichiers de sous-titres temporels (`.srt`).
  
### 2. Neural Machine Translation (NMT)
Nous avons implémenté et comparé **3 approches distinctes** pour traduire le texte francais :
* **Approche 1 (Baseline) :** Un réseau de neurones récurrents **LSTM Seq2Seq** implémenté et entraîné *from scratch* en PyTorch sur ~30 000 phrases.
* **Approche 2 (Transformer Spécialisé) :** Modèles **MarianMT** (`Helsinki-NLP`) spécialisés par paires de langues (fr-en, fr-es).
* **Approche 3 (Modèle Multilingue) :** Modèle **NLLB-200** (`facebook/nllb-200-distilled-600M`) permettant du *Zero-Shot* (traduction vers l'Italien et l'Allemand sans données d'alignement).
  
---
  
## Structure du projet
  
```text
projet_deeplearning/
│
├── data/                       # Dataset mTEDx (Non inclus dans git)
│   ├── fr-fr/                  # Audios (.flac) et transcriptions
│   ├── fr-en/                  # Textes alignés (FR-EN)
│   ├── fr-es/                  # Textes alignés (FR-ES)
│   └── fr-pt/                  # Textes alignés (FR-PT)
│
├── src/                        # Code source (Modulaire)
│   ├── asr_pipeline.py         # Inférence et génération SRT avec Whisper
│   ├── data_loader.py          # Scripts de parsing du format mTEDx (Kaldi)
│   ├── lstm_baseline.py        # Architecture PyTorch et boucle d'entraînement
│   ├── metrics.py              # Calcul WER, CER, BLEU, chrF
│   └── nmt_pipeline.py         # Inférence MarianMT et NLLB-200
│
├── outputs/                    # Résultats générés
│   ├── models/                 # Poids (.pt) du LSTM entraîné
│   └── srt/                    # Fichiers sous-titres finaux (5 langues)
│
├── test.ipynb                  # Notebook principal (Livrable 1)
├── requirements.txt            # Dépendances Python
└── README.md                   # Documentation
```
## Installation et Prérequis
1. Dépendances système

Pour le traitement de l'audio par Hugging Face `transformers`, il est impératif d'installer `ffmpeg` sur votre machine :
- Linux : `sudo apt install ffmpeg`
- MacOS : `brew install ffmpeg`

2. Environnement virtuel

Il est recommandé d'utiliser un environnement virtuel **Poetry** avec Python 3.11+.
```bash
# Cloner le dépôt
git clone <URL_DU_DEPOT>
cd projet_deeplearning
  
### Créer et activer l'environnement
# SI poetry n'est pas installé
pip install poetry

# Puis pour installer toutes les dépendances et l'environnement virtuel
poetry install

# Activer l'environnement virtuel
poetry env activate
```  
## Utilisation
### Notebook Principal

L'évaluation complète, l'entraînement de la baseline et la génération des fichiers se font au sein du Jupyter Notebook

```bash
jupyter notebook test.ipynb
# OU
poetry run jupyter notebook
```

Le Notebook est documenté et divisé en sections correspondant aux attentes du cahier des charges (Chargement, ASR, NMT, Évaluation, et Analyse Qualitative)

### Scripts modulaires

Les pipelines peuvent également être instanciés directement en Python via le dossier `src/` :

```python
from src.asr_pipeline import AudioTranscriber
from src.nmt_pipeline import SubtitleTranslator
  
# 1. Transcription ASR
transcripteur = AudioTranscriber("openai/whisper-small")
texte_fr = transcripteur.transcribe_and_save("audio.flac", "outputs/srt/out.srt")
  
# 2. Traduction NMT
traducteur = SubtitleTranslator("Helsinki-NLP/opus-mt-fr-en")
traducteur.translate_srt("outputs/srt/out.srt", "outputs/srt/out_en.srt")
```

## Résultats Quantitatifs

Évaluation menée sur le set de Test mTEDx (Matériel : NVIDIA RTX 3060 - 6Go VRAM).
1. Performance ASR (Français)

|**Modèle**|**WER**|**CER**|
|---|---|---|
|Whisper-Small|**21.14 %**|**17.42 %**|

2. Performance NMT (Traduction)

|**Modèle**|**Paire**|**Score BLEU**|**Score chrF**|**VRAM**|
|---|---|---|---|---|
|**LSTM Baseline**|FR ➔ EN|1.77|N/A|~0.2 Go|
|**MarianMT**|FR ➔ EN|**42.99**|**64.52**|~0.5 Go|
|**NLLB-200**|FR ➔ EN|39.97|62.64|~2.5 Go|
|**MarianMT**|FR ➔ ES|**49.77**|**69.28**|~0.5 Go|
|**NLLB-200**|FR ➔ ES|44.17|66.38|~2.5 Go|
|**NLLB-200**|FR ➔ PT|**41.84**|**66.53**|~2.5 Go|

## Analyse Qualitative (Cascade Errors)

L'un des défis majeurs de la traduction vidéo est la propagation des erreurs de l'ASR vers le traducteur (Cascade Error)
### Expérience menée :

- Traduction NMT d'un texte français parfait (Ground Truth) : BLEU = 45.30
- Traduction NMT de la sortie générée par Whisper : BLEU = 27.20
- Chute de qualité : -18.1 points BLEU

### Typologie des erreurs identifiées :

1. **Hallucinations ASR :** Whisper invente des mots face au bruit (ex: fusion des mots "pourtant" et "étrangement" en "pourtantrangement"), perturbant totalement le modèle NMT
2. **Entités Nommées** : Transcription phonétique des noms (ex: "Heathcote Williams" devient "Edcott Williams")
3. **Oubli (LSTM)** : L'approche 1 souffre d'évanouissement du gradient sur les longues séquences (>15 mots) contrairement aux Transformers basés sur l'Attention.