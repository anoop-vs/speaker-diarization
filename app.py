import streamlit as st
import os
import nemo.collections.asr as nemo_asr
import numpy as np
from IPython.display import Audio, display
import librosa
import os
import wget
import matplotlib.pyplot as plt

import nemo
import glob
import time

import pprint
pp = pprint.PrettyPrinter(indent=4)

from omegaconf import OmegaConf
import shutil
PAGE_CONFIG = {"page_title":"StColab.io","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)
def main():
	audio = None
	st.title("Awesome Streamlit for ML")
	st.subheader("How to run streamlit from colab")
	menu = ["Home","About"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Home':
		audio = st.file_uploader("Upload Audio", type=["wav"])
		if audio is not None:
			file_details = {"filename":audio.name, "filetype":audio.type, "filesize":audio.size}
			st.text("File name: " + file_details["filename"])
			st.text("File type: "  + file_details["filetype"])
			st.text("File type: "  + str(file_details["filesize"]))
			with open(file_details["filename"],"wb") as f:
				f.write(audio.getbuffer())
			#st.button ("Submit")
			st.audio(audio, format="audio/wav", start_time=0)
			with st.spinner('Wait for it...'):
				time.sleep(20)
			CONFIG_URL = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/offline_diarization_with_asr.yaml"

			if not os.path.exists('offline_diarization_with_asr.yaml'):
				CONFIG = wget.download(CONFIG_URL)
			else:
				CONFIG = 'offline_diarization_with_asr.yaml'

			cfg = OmegaConf.load(CONFIG)
			#st.text(OmegaConf.to_yaml(cfg))

			import json
			meta = {'audio_filepath': file_details["filename"], 'offset': 0, 'duration':None, 'label': 'infer', 'text': '-', 'num_speakers': 2, 'rttm_filepath': None, 'uem_filepath' : None}
			with open(os.path.join('input_manifest.json'),'w') as fp:
				json.dump(meta,fp)
				fp.write('\n')
			cfg.diarizer.manifest_filepath = 'input_manifest.json'
			pretrained_speaker_model='titanet_large'
			cfg.diarizer.manifest_filepath = cfg.diarizer.manifest_filepath
			cfg.diarizer.out_dir = "/" #Directory to store intermediate files and prediction outputs
			cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
			cfg.diarizer.speaker_embeddings.parameters.window_length_in_sec = 1.5
			cfg.diarizer.speaker_embeddings.parameters.shift_length_in_sec = 0.75
			cfg.diarizer.clustering.parameters.oracle_num_speakers=True

			# Using VAD generated from ASR timestamps
			cfg.diarizer.asr.model_path = 'QuartzNet15x5Base-En'
			cfg.diarizer.oracle_vad = False # ----> Not using oracle VAD 
			cfg.diarizer.asr.parameters.asr_based_vad = True
			cfg.diarizer.asr.parameters.threshold=100 # ASR based VAD threshold: If 100, all silences under 1 sec are ignored.
			cfg.diarizer.asr.parameters.decoder_delay_in_sec=0.2 # Decoder delay is compensated for 0.2 sec

			from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASR_TIMESTAMPS
			asr_ts_decoder = ASR_TIMESTAMPS(**cfg.diarizer)
			asr_model = asr_ts_decoder.set_asr_model()
			word_hyp, word_ts_hyp = asr_ts_decoder.run_ASR(asr_model)

			st.text("Decoded word output dictionary: " + str(word_hyp['commercial_mono']))
			st.text("Word-level timestamps dictionary: " + str(word_ts_hyp['commercial_mono']))
	 
			from nemo.collections.asr.parts.utils.diarization_utils import ASR_DIAR_OFFLINE
			asr_diar_offline = ASR_DIAR_OFFLINE(**cfg.diarizer)
			asr_diar_offline.word_ts_anchor_offset = asr_ts_decoder.word_ts_anchor_offset

			diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
			st.text("Diarization hypothesis output: \n" + str(diar_hyp['commercial_mono']))
	 
			def read_file(path_to_file):
				with open(path_to_file) as f:
					contents = f.read().splitlines()
				return contents

			predicted_speaker_label_rttm_path = f"/pred_rttms/commercial_mono.rttm"
			pred_rttm = read_file(predicted_speaker_label_rttm_path)

			st.text(pred_rttm)

			from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
			pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)
	 
			asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
	 
			transcription_path_to_file = f"/pred_rttms/commercial_mono.txt"
			transcript = read_file(transcription_path_to_file)
			st.text(transcript)
	 
			transcription_path_to_file = f"/pred_rttms/commercial_mono.json"
			json_contents = read_file(transcription_path_to_file)
			st.json(json_contents)

if __name__ == '__main__':
	main()