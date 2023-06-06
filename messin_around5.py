from diffusers import Mel
import torch
import torchaudio
import torchaudio.transforms as T
import torchvision.transforms as tvt
from PIL import Image

output_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\00001-f000009.wav"
output_data_root = 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\EN\\9\\sentence_level_audio\\'

waveform, sr = torchaudio.load(output_file)

# Downsampled
resampler = T.Resample(44100, 22050, dtype=waveform.dtype)
resampled_waveform = resampler(waveform)
sr = 22050
waveform = resampled_waveform

# Get mel_spec
image_size = 256
MEL = Mel(x_res=image_size, y_res=image_size) # create MEL object, has variety of methods
img_to_tensor = tvt.PILToTensor() # create PILToTensor transform.

MEL.load_audio(raw_audio=waveform[0]) # load waveform which has shape [[N_samples, ]]
spectros = []
for slice in range(MEL.get_number_of_slices()):  # MEL slices audio depending on size of x_res, y_res
    spectro = MEL.audio_slice_to_image(slice)  # convert to image
    spectro = img_to_tensor(spectro) / 255.0  # convert to tensor and normalize
    # print(spectro.shape)
    # plt.imshow(spectro[0])
    # plt.show()
    # input("continue")
    spectros.append(spectro)
spectros = torch.stack(spectros)

# Plot
print(spectros.squeeze(0)[0].shape)
im = Image.fromarray(spectros.squeeze(1).numpy()[0] * 255).convert('L')  # extract sample and rescale it
im.save(f"Output/diff_spec.jpg")

# Get back audio
audio = torch.tensor([MEL.image_to_audio(im)])  # allows torch audio to save file
torchaudio.save(f"Output/diff_audio.wav", audio, 22500)  # save new file


input_data_root = 'C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Data\\LibriS2S\\DE\\9\\sentence_level_audio\\'

