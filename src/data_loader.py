import os
from datasets import Dataset, DatasetDict

def load_mtedx_data(data_dir, pairs=['fr-en', 'fr-es']):
    """
    Charge les données mTEDx en lisant directement les fichiers textes alignés.
    Idéal pour la tâche de Traduction Neuronale (NMT).
    """
    datasets = {}

    for pair in pairs:
        # Ex: src_lang='fr', tgt_lang='en'
        src_lang, tgt_lang = pair.split('-')
        pair_dir = os.path.join(data_dir, pair, 'data')
        
        if not os.path.exists(pair_dir):
            print(f"Dossier introuvable pour la paire {pair} : {pair_dir}")
            continue

        dataset_splits = {}
        for split in ['train', 'valid', 'test']:
            txt_dir = os.path.join(pair_dir, split, 'txt')
            
            # Dans mTEDx, les fichiers s'appellent train.fr, train.en, etc.
            src_file = os.path.join(txt_dir, f"{split}.{src_lang}")
            tgt_file = os.path.join(txt_dir, f"{split}.{tgt_lang}")

            if os.path.exists(src_file) and os.path.exists(tgt_file):
                # Lire les fichiers ligne par ligne
                with open(src_file, 'r', encoding='utf-8') as f_src, \
                     open(tgt_file, 'r', encoding='utf-8') as f_tgt:
                    
                    src_lines = [line.strip() for line in f_src]
                    tgt_lines = [line.strip() for line in f_tgt]

                # Vérifier l'alignement (1 phrase FR = 1 phrase EN)
                if len(src_lines) == len(tgt_lines):
                    # Créer une liste de dictionnaires
                    data = [{'src_text': s, 'tgt_text': t} for s, t in zip(src_lines, tgt_lines)]
                    # Convertir au format Hugging Face Dataset
                    dataset_splits[split] = Dataset.from_list(data)
                    print(f"{pair} - {split} chargé : {len(data)} phrases.")
                else:
                    print(f"Erreur d'alignement dans {pair}/{split} ({len(src_lines)} vs {len(tgt_lines)} lignes)")
            else:
                pass # On ignore silencieusement si les fichiers n'existent pas

        if dataset_splits:
            datasets[pair] = DatasetDict(dataset_splits)

    return datasets