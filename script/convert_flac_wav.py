from pydub import AudioSegment
import torchaudio
import os
import glob

input_folder = "/home/wxzhang/projects/coding4paper/projects/subband/data/123859"
output_folder = os.path.join("/home/wxzhang/projects/coding4paper/projects/subband/data", "wav_set_1")

os.makedirs(output_folder, exist_ok=True)
input_path_list = glob.glob(os.path.join(input_folder, '*.flac'))

for file_path in input_path_list:
    file_name = os.path.basename(file_path).split(".")[0]
    audio_file = AudioSegment.from_file(file_path, format='flac')
    cur_output_path = os.path.join(output_folder, file_name+".wav")
    audio_file.export(cur_output_path, format='wav')
