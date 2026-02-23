import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_mtedx_data

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

        # Setup structure for fr-en (Kaldi style with segments)
        # data/fr-en/data/train/
        #   segments
        #   wav.scp
        #   txt/train.fr
        #   txt/train.en
        #   wav/audio.wav

        self.pair_dir = os.path.join(self.test_dir, 'fr-en')
        self.data_dir = os.path.join(self.pair_dir, 'data', 'train')
        self.txt_dir = os.path.join(self.data_dir, 'txt')
        self.wav_dir = os.path.join(self.data_dir, 'wav')

        os.makedirs(self.txt_dir, exist_ok=True)
        os.makedirs(self.wav_dir, exist_ok=True)

        # Create dummy files
        # 1. wav file
        self.wav_path = os.path.join(self.wav_dir, 'audio.wav')
        with open(self.wav_path, 'wb') as f:
            f.write(b'RIFF....WAVEfmt ....data....') # Dummy wav header

        # 2. segments
        # segment_id recording_id start end
        segments_content = "seg1 audio 0.0 1.0\nseg2 audio 1.0 2.0"
        with open(os.path.join(self.data_dir, 'segments'), 'w') as f:
            f.write(segments_content)

        # 3. wav.scp
        # recording_id path
        wav_scp_content = f"audio {self.wav_path}"
        with open(os.path.join(self.data_dir, 'wav.scp'), 'w') as f:
            f.write(wav_scp_content)

        # 4. text files
        # Assuming line-by-line corresponding to segments
        src_text = "Bonjour le monde\nComment allez-vous"
        tgt_text = "Hello world\nHow are you"

        # In the loader, it looks for {split}.{lang} inside txt folder
        # split is 'train'
        with open(os.path.join(self.txt_dir, 'train.fr'), 'w') as f:
            f.write(src_text)
        with open(os.path.join(self.txt_dir, 'train.en'), 'w') as f:
            f.write(tgt_text)

        # Setup structure for fr-es (Simple folders, no segments)
        # data/fr-es/data/train/
        #   wav/file1.wav
        #   txt/file1.fr
        #   txt/file1.es
        self.pair_es_dir = os.path.join(self.test_dir, 'fr-es')
        self.es_data_dir = os.path.join(self.pair_es_dir, 'data', 'train')
        self.es_txt_dir = os.path.join(self.es_data_dir, 'txt')
        self.es_wav_dir = os.path.join(self.es_data_dir, 'wav')

        os.makedirs(self.es_txt_dir, exist_ok=True)
        os.makedirs(self.es_wav_dir, exist_ok=True)

        with open(os.path.join(self.es_wav_dir, 'file1.wav'), 'wb') as f:
            f.write(b'dummy')
        with open(os.path.join(self.es_txt_dir, 'file1.fr'), 'w') as f:
            f.write("Salut")
        with open(os.path.join(self.es_txt_dir, 'file1.es'), 'w') as f:
            f.write("Hola")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_load_mtedx_data(self):
        datasets = load_mtedx_data(self.test_dir, pairs=['fr-en', 'fr-es'])

        # Test fr-en (Segments based)
        self.assertIn('fr-en', datasets)
        self.assertIn('train', datasets['fr-en'])

        ds = datasets['fr-en']['train']
        self.assertEqual(len(ds), 2)

        item = ds[0]
        self.assertEqual(item['id'], 'seg1')
        self.assertEqual(item['src_text'], 'Bonjour le monde')
        self.assertEqual(item['tgt_text'], 'Hello world')
        self.assertEqual(item['audio_path'], self.wav_path)
        self.assertEqual(item['start'], 0.0)
        self.assertEqual(item['end'], 1.0)

        # Test fr-es (Folder based)
        self.assertIn('fr-es', datasets)
        self.assertIn('train', datasets['fr-es'])

        ds_es = datasets['fr-es']['train']
        self.assertEqual(len(ds_es), 1)

        item_es = ds_es[0]
        self.assertEqual(item_es['id'], 'file1')
        self.assertEqual(item_es['src_text'], 'Salut')
        self.assertEqual(item_es['tgt_text'], 'Hola')
        self.assertEqual(item_es['audio_path'], os.path.join(self.es_wav_dir, 'file1.wav'))

if __name__ == '__main__':
    unittest.main()
