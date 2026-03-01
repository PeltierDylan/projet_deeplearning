import os
import torch
from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer

class SubtitleTranslator:
    def __init__(self, model_name="Helsinki-NLP/opus-mt-fr-en"):
        """
        Initialise le modèle de traduction.
        Par défaut, on utilise MarianMT pour la traduction Français -> Anglais.
        Pour l'espagnol, on utilisera "Helsinki-NLP/opus-mt-fr-es".
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Chargement du modèle NMT '{model_name}' sur : {self.device}...")
        
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(self.device)

    def translate_text(self, text):
        """Traduit une simple phrase ou une liste de phrases."""
        # Préparation du texte pour le modèle
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # Génération de la traduction
        translated_tokens = self.model.generate(**inputs)
        
        # Décodage des tokens en texte lisible
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0] if isinstance(text, str) else translated_text

    def translate_srt(self, input_srt_path, output_srt_path):
        """Lit un fichier SRT en français et génère un SRT traduit."""
        print(f"Traduction du fichier '{input_srt_path}' en cours...")
        
        with open(input_srt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        translated_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Si c'est un numéro de bloc ou un timestamp (ex: 00:00:01,000 --> 00:00:03,000)
            if line.isdigit() or "-->" in line:
                translated_lines.append(line + "\n")
                i += 1
            # Si c'est du texte à traduire
            elif line != "":
                # On regroupe le texte s'il est sur plusieurs lignes
                text_block = line
                while i + 1 < len(lines) and lines[i+1].strip() != "" and not lines[i+1].strip().isdigit():
                    i += 1
                    text_block += " " + lines[i].strip()
                
                # Traduction
                traduction = self.translate_text(text_block)
                translated_lines.append(traduction + "\n")
                i += 1
            # Ligne vide (séparation des blocs)
            else:
                translated_lines.append("\n")
                i += 1

        # Sauvegarde du nouveau fichier SRT
        os.makedirs(os.path.dirname(output_srt_path), exist_ok=True)
        with open(output_srt_path, 'w', encoding='utf-8') as f:
            f.writelines(translated_lines)
            
        print(f"✅ Fichier SRT traduit sauvegardé sous : {output_srt_path}")

class MultilingualTranslator:
    def __init__(self, model_name="facebook/nllb-200-distilled-600M", src_lang="fra_Latn"):
        """
        Initialise le modèle multilingue NLLB.
        src_lang doit être au format BCP-47 (ex: fra_Latn pour le français)
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Chargement du modèle Multilingue '{model_name}' sur : {self.device}...")
        
        # Chargement du tokenizer et du modèle
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate_text(self, text, tgt_lang="eng_Latn"):
        """
        Traduit le texte vers la langue cible spécifiée.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        # CORRECTIF ICI : Utilisation de convert_tokens_to_ids au lieu de lang_code_to_id
        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang)
        
        translated_tokens = self.model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=150
        )
        
        translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0]