import os
import torch
from transformers import pipeline

def format_timestamp(seconds):
    """Convertit des secondes (float) au format temporel SRT (HH:MM:SS,mmm)"""
    if seconds is None:
        return "00:00:00,000"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def write_srt(chunks, output_path):
    """Écrit les segments de transcription dans un fichier .srt"""
    # Créer le dossier parent si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            start, end = chunk['timestamp']
            
            # Sécurité : si le modèle ne trouve pas de timestamp de fin
            if end is None:
                end = start + 2.0 
            
            start_str = format_timestamp(start)
            end_str = format_timestamp(end)
            text = chunk['text'].strip()
            
            # Format standard SRT
            f.write(f"{i + 1}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{text}\n\n")

class AudioTranscriber:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialise le pipeline ASR. 
        On utilise whisper-small pour avoir un bon compromis vitesse/qualité.
        """
        # Vérifie si une carte graphique (GPU) est disponible, sinon utilise le processeur (CPU)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Chargement du modèle ASR '{model_name}' sur : {self.device}...")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
            chunk_length_s=30, # Permet de traiter des audios longs en les découpant
            return_timestamps=True # Indispensable pour générer le SRT !
        )

    def transcribe_and_save(self, audio_path, output_srt_path):
        """Transcripte l'audio et sauvegarde le résultat en .srt"""
        print(f"Transcription de '{audio_path}' en cours...")
        
        # On force la langue en français pour aider le modèle à ne pas se tromper
        result = self.pipe(audio_path, generate_kwargs={"language": "french", "task": "transcribe"})
        
        # Génération du fichier SRT
        write_srt(result["chunks"], output_srt_path)
        print(f"Terminé ! Fichier SRT sauvegardé sous : {output_srt_path}")
        
        return result["text"]