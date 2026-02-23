import os
import glob
from pathlib import Path
from datasets import Dataset, DatasetDict, Audio

def load_kaldi_format(data_dir, split, src_lang, tgt_lang):
    """
    Load data from a Kaldi-style directory structure commonly found in OpenSLR datasets.
    Expected files: segments, wav.scp, text (or similar)
    """
    split_dir = os.path.join(data_dir, split)

    # Check for segments file
    segments_file = os.path.join(split_dir, 'segments')
    wav_scp_file = os.path.join(split_dir, 'wav.scp')
    text_file = os.path.join(split_dir, 'text') # May be different name

    # If standard Kaldi files are missing, try to find them by pattern or look for txt/wav folders
    # OpenSLR mTEDx structure often has:
    # data/train/txt/train.fr (text)
    # data/train/wav/train.wav (audio) -> This implies one long audio file per split?
    # Or maybe many wav files.

    # Let's try to handle the specific case of mTEDx where it might be structured as:
    # data/train/txt/train.fr
    # data/train/txt/train.en
    # data/train/wav/train.wav (if single file) or multiple files
    # data/train/segments (if present)

    segments_path = Path(split_dir) / 'segments'
    if segments_path.exists():
        return _load_from_segments(split_dir, src_lang, tgt_lang)

    # Fallback: look for individual wav files and parallel text files
    wav_dir = Path(split_dir) / 'wav'
    txt_dir = Path(split_dir) / 'txt'

    if wav_dir.exists() and txt_dir.exists():
        return _load_from_folders(wav_dir, txt_dir, src_lang, tgt_lang)

    print(f"Warning: Could not find data for split {split} in {data_dir}")
    return None

def _load_from_segments(split_dir, src_lang, tgt_lang):
    """
    Load data using segments file (Kaldi style).
    """
    segments_file = os.path.join(split_dir, 'segments')
    wav_scp_file = os.path.join(split_dir, 'wav.scp')

    # Read wav.scp to get recording_id -> wav_path mapping
    wav_paths = {}
    if os.path.exists(wav_scp_file):
        with open(wav_scp_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    wav_paths[parts[0]] = parts[1]

    # Read segments to get segment_id -> (recording_id, start, end)
    segments = []
    with open(segments_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                segments.append({
                    'segment_id': parts[0],
                    'recording_id': parts[1],
                    'start': float(parts[2]),
                    'end': float(parts[3])
                })

    # Read text files (assuming name format like train.fr, train.en or similar in 'txt' folder or root)
    # Usually in mTEDx, they are in 'txt' folder
    src_text_file = os.path.join(split_dir, 'txt', f'{os.path.basename(split_dir)}.{src_lang}')
    tgt_text_file = os.path.join(split_dir, 'txt', f'{os.path.basename(split_dir)}.{tgt_lang}')

    if not os.path.exists(src_text_file):
        # Try looking in root of split_dir
        src_text_file = os.path.join(split_dir, f'{os.path.basename(split_dir)}.{src_lang}')
        tgt_text_file = os.path.join(split_dir, f'{os.path.basename(split_dir)}.{tgt_lang}')

    src_texts = {}
    tgt_texts = {}

    # Helper to read text file which might be id-based or line-based corresponding to segments
    # mTEDx usually aligns line-by-line with segments if there are no IDs in text file.
    # But let's check if it has IDs.

    def read_text_file(filepath):
        texts = {}
        lines = []
        with open(filepath, 'r') as f:
            for line in f:
                lines.append(line.strip())
        return lines

    if os.path.exists(src_text_file):
        src_lines = read_text_file(src_text_file)
    else:
        print(f"Warning: Source text file not found: {src_text_file}")
        src_lines = []

    if os.path.exists(tgt_text_file):
        tgt_lines = read_text_file(tgt_text_file)
    else:
        print(f"Warning: Target text file not found: {tgt_text_file}")
        tgt_lines = []

    # Combine
    data = []
    # If text lines match segments count, assume 1-to-1 mapping
    if len(segments) == len(src_lines):
        for i, seg in enumerate(segments):
            rec_id = seg['recording_id']
            if rec_id in wav_paths:
                audio_path = wav_paths[rec_id]
                # We need to extract the segment from the audio file using start/end times
                # But Dataset.from_dict usually expects full audio files.
                # For efficiency, we can pass the audio path and segment info,
                # and let a transform handle it, or use ffmpeg to slice (which is slow).
                # Here we will store metadata and let the pipeline handle slicing if needed,
                # or use 'audio' feature with 'sampling_rate' if loading raw audio.

                # For simplicity in this loader, we return the path and segment times.

                entry = {
                    'id': seg['segment_id'],
                    'audio_path': audio_path,
                    'start': seg['start'],
                    'end': seg['end'],
                    'src_text': src_lines[i],
                    'tgt_text': tgt_lines[i] if i < len(tgt_lines) else ""
                }
                data.append(entry)
    else:
        print(f"Warning: Mismatch in segments ({len(segments)}) and text lines ({len(src_lines)}) for {split_dir}")

    return data

def _load_from_folders(wav_dir, txt_dir, src_lang, tgt_lang):
    """
    Load from wav and txt folders where filenames match.
    """
    wav_files = sorted(list(wav_dir.glob('*.wav')))
    data = []

    for wav_file in wav_files:
        file_id = wav_file.stem
        src_txt_path = txt_dir / f"{file_id}.{src_lang}"
        tgt_txt_path = txt_dir / f"{file_id}.{tgt_lang}"

        src_text = ""
        tgt_text = ""

        if src_txt_path.exists():
            with open(src_txt_path, 'r') as f:
                src_text = f.read().strip()

        if tgt_txt_path.exists():
            with open(tgt_txt_path, 'r') as f:
                tgt_text = f.read().strip()

        data.append({
            'id': file_id,
            'audio_path': str(wav_file),
            'start': 0.0,
            'end': None, # Full file
            'src_text': src_text,
            'tgt_text': tgt_text
        })

    return data

def load_mtedx_data(data_dir, pairs=['fr-en', 'fr-es']):
    """
    Load Multilingual TEDx dataset for specified language pairs.

    Args:
        data_dir (str): Path to the root of the extracted dataset.
                        Expected structure: data_dir/fr-en/data/...
        pairs (list): List of language pairs to load (e.g., ['fr-en', 'fr-es']).

    Returns:
        dict: A dictionary where keys are language pairs and values are DatasetDict objects.
    """
    datasets = {}

    for pair in pairs:
        pair_dir = os.path.join(data_dir, pair)
        if not os.path.exists(pair_dir):
            print(f"Warning: Directory for pair {pair} not found at {pair_dir}")
            continue

        src_lang, tgt_lang = pair.split('-')

        # Check for 'data' subdirectory which is common in OpenSLR structures
        data_subdir = os.path.join(pair_dir, 'data')
        if not os.path.exists(data_subdir):
            data_subdir = pair_dir # Fallback if data is directly in pair_dir

        splits = ['train', 'valid', 'test']
        dataset_splits = {}

        for split in splits:
            # Handle potential naming differences (valid vs dev)
            split_name = split
            split_path = os.path.join(data_subdir, split)
            if not os.path.exists(split_path):
                if split == 'valid':
                    split_path = os.path.join(data_subdir, 'dev')
                    if os.path.exists(split_path):
                        split_name = 'dev'
                    else:
                        split_path = os.path.join(data_subdir, 'val')
                        if os.path.exists(split_path):
                            split_name = 'val'

            if os.path.exists(split_path):
                print(f"Loading {split} split for {pair} from {split_path}...")
                data = load_kaldi_format(data_subdir, split_name, src_lang, tgt_lang)
                if data:
                    # Create Hugging Face Dataset
                    hf_dataset = Dataset.from_list(data)
                    # Cast audio column if needed (but we have paths and segments,
                    # generic Audio() feature requires a single file per example usually.
                    # We will keep paths for now and let the user handle loading/slicing
                    # or add a transform later).
                    dataset_splits[split] = hf_dataset
            else:
                print(f"Warning: Split {split} not found for {pair}")

        if dataset_splits:
            datasets[pair] = DatasetDict(dataset_splits)

    return datasets

if __name__ == "__main__":
    # Example usage (commented out)
    # data = load_mtedx_data("/path/to/mtedx", pairs=['fr-en'])
    # print(data)
    pass
