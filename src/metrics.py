import jiwer
import sacrebleu
import re

def clean_text_for_asr(text):
    """
    Nettoie le texte pour l'évaluation ASR (minuscules, retrait de la ponctuation).
    L'ASR est souvent évalué sur les mots purs, sans tenir compte de la casse.
    """
    text = text.lower()
    # Retire la ponctuation de base pour ne garder que les mots et les espaces
    text = re.sub(r'[^\w\s\']', '', text)
    # Remplace les sauts de ligne par des espaces
    text = text.replace('\n', ' ')
    return " ".join(text.split())

def evaluate_asr(reference_texts, hypothesis_texts):
    """
    Calcule le WER et le CER pour l'ASR en utilisant jiwer.
    Accepte soit une chaîne de caractères (1 fichier complet), soit une liste de phrases.
    """
    if isinstance(reference_texts, str):
        reference_texts = [reference_texts]
        hypothesis_texts = [hypothesis_texts]

    refs_clean = [clean_text_for_asr(r) for r in reference_texts]
    hyps_clean = [clean_text_for_asr(h) for h in hypothesis_texts]
    
    wer = jiwer.wer(refs_clean, hyps_clean)
    cer = jiwer.cer(refs_clean, hyps_clean)
    
    # On retourne les scores en pourcentage pour plus de lisibilité
    return {"WER": round(wer * 100, 2), "CER": round(cer * 100, 2)}

def evaluate_nmt(references, hypotheses):
    """
    Calcule BLEU et chrF pour la traduction en utilisant sacrebleu.
    Attend une liste de phrases de référence et une liste de phrases générées.
    """
    # Sacrebleu attend une liste de listes pour les références (cas où il y a plusieurs trads possibles)
    refs_list = [references] 
    
    bleu = sacrebleu.corpus_bleu(hypotheses, refs_list)
    chrf = sacrebleu.corpus_chrf(hypotheses, refs_list)
    
    return {"BLEU": round(bleu.score, 2), "chrF": round(chrf.score, 2)}

def extract_text_from_srt(srt_path):
    """Utilitaire pour extraire uniquement le texte brut d'un fichier .srt"""
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    text_blocks = []
    for line in lines:
        line = line.strip()
        # On ignore les numéros de séquence, les timestamps et les lignes vides
        if not line.isdigit() and "-->" not in line and line != "":
            text_blocks.append(line)
            
    return " ".join(text_blocks)