# Team Members : Mortiniera Thevie, Karthigan Sinnathamby, Mohamed Ndoye, Frédéric Myotte

# text_transfert_over_audio_channel
Develop a proof of concept to show that we can exchange files “over the air” between two laptops, by using the speaker of one laptop as the transmitting device and the microphone of the other laptop as the receiving device in presence of an interfering third party.

# Requirements 

The projects requires the following python libraries : 

numpy as np
pyaudio
struct
matplotlib
scipy
soundfile
sounddevice

# Running 

The test were executed as follows  :
- Create a 160 characters text file(or use the already create text file "test160.txt")
- Generate the gaussian noise (1-2khz or 2-3khz) . The files are in the folder.
- Run : "python emitter.py test160.txt"
- Run : "receiver.py"

The emitter should encode the text file as sound, while the receiver should recognize by himself the non-blocking frequency range of the generated noise and decode the receiving sound as the output textfile, all in 3 minutes. It prints the decoded text in the console.
