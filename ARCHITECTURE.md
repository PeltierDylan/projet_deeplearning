# Multilingual Video Translation Pipeline

This document outlines the architecture for the multilingual video translation project using the Multilingual TEDx dataset.

## Pipeline Overview

The pipeline consists of three main stages:
1.  **Audio Extraction & Segmentation**
2.  **Automatic Speech Recognition (ASR)**
3.  **Neural Machine Translation (NMT)**

### 1. Audio Extraction
- **Input**: Video files (mp4, mkv) or long Audio files (wav).
- **Tools**: `ffmpeg-python` for extraction and conversion.
- **Process**:
    - Extract audio track from video.
    - Convert to 16kHz mono WAV format (standard for most ASR models).
    - Segment audio based on `segments` file (start/end timestamps) if using the mTEDx format.

### 2. Automatic Speech Recognition (ASR)
- **Goal**: Transcribe French audio to French text.
- **Models**:
    - **Wav2Vec2**: `facebook/wav2vec2-large-xlsr-53-french` or similar fine-tuned models.
    - **Whisper**: `openai/whisper-small` or `medium`.
- **Output**: French transcripts.

### 3. Neural Machine Translation (NMT)
- **Goal**: Translate French transcripts to Target Languages (English, Spanish, etc.).
- **Models**:
    - **Seq2Seq LSTM**: Baseline (optional).
    - **Transformer (MarianMT)**: `Helsinki-NLP/opus-mt-fr-en`, `Helsinki-NLP/opus-mt-fr-es`.
    - **Massively Multilingual (NLLB)**: `facebook/nllb-200-distilled-600M`.
- **Output**: Translated text.

---

## Handling Cascade Errors

"Cascade errors" occur when mistakes made by the ASR system propagate to the NMT system, leading to poor translations. Here are strategies to mitigate them:

### 1. Confidence Filtering
- **Concept**: ASR models often provide a confidence score or probability for the transcribed text.
- **Implementation**:
    - Calculate the average log-probability of the ASR output tokens.
    - If the confidence is below a certain threshold (e.g., 0.6), flag the segment for manual review or discard it to prevent "hallucinated" translations.

### 2. N-best Lists
- **Concept**: Instead of taking only the single best ASR hypothesis (beam size 1), take the top-N hypotheses (e.g., N=5).
- **Implementation**:
    - Feed all N hypotheses into the NMT model.
    - Use a re-ranking mechanism (e.g., using a language model or back-translation) to select the best final translation.
    - Alternatively, concatenate hypotheses or use a "lattice-to-sequence" model if supported.

### 3. Robust NMT Training (Data Augmentation)
- **Concept**: Train the NMT model to be robust to ASR errors.
- **Implementation**:
    - **Simulated Errors**: During NMT training, inject noise into the source (French) text (e.g., random deletions, substitutions) to mimic ASR errors.
    - **ASR Output Training**: If you have ground truth translations, transcribe the training audio using your ASR model and train the NMT model on `(ASR_Output, Ground_Truth_Translation)` pairs instead of `(Gold_Transcript, Ground_Truth_Translation)`. This adapts the NMT model to the specific error patterns of the ASR.

### 4. End-to-End Models (Alternative)
- **Concept**: Use models that go directly from Audio to Translated Text (Speech-to-Text Translation, ST).
- **Models**: `facebook/seamless-m4t`, `openai/whisper` (task='translate').
- **Note**: While this avoids cascade errors, the project requirements explicitly ask for separate ASR and NMT modules for comparison. However, comparing an end-to-end model against the cascade is a valuable evaluation metric.
