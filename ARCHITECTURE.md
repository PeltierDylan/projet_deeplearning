# Pipeline de Traduction Vidéo Multilingue

Ce document décrit l'architecture du projet de traduction de vidéos multilingues utilisant l'ensemble de données TEDx multilingue.

## Aperçu du Pipeline

Le pipeline comprend trois étapes principales:
1.  **Extraction Audio et Segmentation**
2.  **Reconnaissance Automatique de la Parole (ASR)**
3.  **Traduction Automatique Neuronale (NMT)**

### 1. Extraction Audio
- **Entrée**: Fichiers vidéo (mp4, mkv) ou fichiers audio longs (wav).
- **Outils**: `ffmpeg-python` pour l'extraction et la conversion.
- **Processus**:
    - Extraire la piste audio de la vidéo.
    - Convertir au format WAV mono 16 kHz (standard pour la plupart des modèles ASR).
    - Segmenter l'audio en fonction du fichier `segments` (horodatages début/fin) si on utilise le format mTEDx.

### 2. Reconnaissance Automatique de la Parole (ASR)
- **Objectif**: Transcrire l'audio français en texte français.
- **Modèles**:
    - **Wav2Vec2**: `facebook/wav2vec2-large-xlsr-53-french` ou modèles similaires affinés.
    - **Whisper**: `openai/whisper-small` ou `medium`.
- **Sortie**: Transcripts français.

### 3. Traduction Automatique Neuronale (NMT)
- **Objectif**: Traduire les transcripts français en langues cibles (anglais, espagnol, etc.).
- **Modèles**:
    - **Seq2Seq LSTM**: Baseline (optionnel).
    - **Transformer (MarianMT)**: `Helsinki-NLP/opus-mt-fr-en`, `Helsinki-NLP/opus-mt-fr-es`.
    - **Multilingue Massif (NLLB)**: `facebook/nllb-200-distilled-600M`.
- **Sortie**: Texte traduit.

---

## Gestion des Erreurs en Cascade

Les « erreurs en cascade » se produisent lorsque les erreurs commises par le système ASR se propagent au système NMT, conduisant à de mauvaises traductions. Voici des stratégies pour les atténuer:

### 1. Filtrage de Confiance
- **Concept**: Les modèles ASR fournissent souvent un score de confiance ou une probabilité pour le texte transcrit.
- **Implémentation**:
    - Calculer la log-probabilité moyenne des tokens de sortie ASR.
    - Si la confiance est en dessous d'un certain seuil (par ex. 0.6), marquer le segment pour révision manuelle ou le rejeter pour éviter les traductions « hallucées ».

### 2. Listes N-meilleures
- **Concept**: Au lieu de prendre seulement la meilleure hypothèse ASR unique (beam size 1), prendre les N meilleures hypothèses (par ex. N=5).
- **Implémentation**:
    - Alimenter toutes les N hypothèses dans le modèle NMT.
    - Utiliser un mécanisme de ré-classement (par ex. en utilisant un modèle de langue ou une rétro-traduction) pour sélectionner la meilleure traduction finale.
    - Alternativement, concaténer les hypothèses ou utiliser un modèle « lattice-to-sequence » si supporté.

### 3. Entraînement NMT Robuste (Augmentation de Données)
- **Concept**: Entraîner le modèle NMT pour être robuste aux erreurs ASR.
- **Implémentation**:
    - **Erreurs Simulées**: Pendant l'entraînement NMT, injecter du bruit dans le texte source (français) (par ex. suppressions aléatoires, substitutions) pour mimer les erreurs ASR.
    - **Entraînement de Sortie ASR**: Si vous avez des traductions de vérité de base, transcrire l'audio d'entraînement en utilisant votre modèle ASR et entraîner le modèle NMT sur des paires `(ASR_Output, Ground_Truth_Translation)` au lieu de `(Gold_Transcript, Ground_Truth_Translation)`. Cela adapte le modèle NMT aux schémas d'erreurs spécifiques de l'ASR.

### 4. Modèles Bout à Bout (Alternative)
- **Concept**: Utiliser des modèles qui vont directement de l'Audio au Texte Traduit (Speech-to-Text Translation, ST).
- **Modèles**: `facebook/seamless-m4t`, `openai/whisper` (task='translate').
- **Note**: Bien que cela évite les erreurs en cascade, les exigences du projet demandent explicitement des modules ASR et NMT séparés pour la comparaison. Cependant, comparer un modèle bout à bout contre la cascade est une métrique d'évaluation précieuse.
