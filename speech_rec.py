import speech_recognition as sr


def STT(outfile, AUDIO_FILE="transcript.wav"):
    """Speech Recognition: Transcribes audio file"""

    # use the audio file as the audio source
    r = sr.Recognizer()
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)  # read the entire audio file

        output = "Transcription: " + r.recognize_google(audio)

        print(output)
    with open(outfile, 'w', encoding='utf-8') as myfile:
        myfile.write(output)


if __name__ == "__main__":
    outfile = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Demo\\00001-f000001_synth_asr.txt"
    audio_file = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Demo\\00001-f000001.wav"

    STT(outfile, audio_file)

    pass