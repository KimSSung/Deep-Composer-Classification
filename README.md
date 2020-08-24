# Style Transfer using Symbolic Music (MIDI)
Style Transfer between different composer styles using symbolic representation (MIDI) of music, implemented in PyTorch. 

## Abstract
This research has its purpose in accelerating the usage of symbolic representation of music through showing its potential in improving accuracy and efficiency in recognizing and handling musical information.
Classification task, whether it's concerning genre, style, era, or composer as each category, is a relatively common in the music domain. To the best of our knowledge, all the previous works have been mostly attempted using audio/wav features, such as MFCC, mel-spectrogram. However, using only audio/wav format to detect, analyze and maintain music has its limitations. Through this research, we show that using symbolic representation(Midi, music piece, etc) can strengthen certain aspects of music detection and classification, which is often the basis of many tasks in music industry, and suggest symbolic representation as a possible solution to the current limitations and problems present in working with audio/wav format. Furthermore, we went ahead and tested serveral adversarial attack techniques on this format of music. 

## Requirements
#### Library
> * Python >= 3.6.9
> * PyTorch = 1.4.0
> * torchvision = 
> * pymidi_csv = 1.10.0
> * numpy = 1.18.1
> * sklearn = 
> * matplotlib = 3.1.3
> * music21 = 5.7.2
> * tqdm

#### Dataset
[MAESTRO][maestro_link]: (MIDI and Audio Edited for Synchronous TRacks and Organization) is a dataset composed of over 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.     

Specifically, we used **v2.0.0** version of the dataset. Although the big advantage of using this dataset is the fine alignment between midi & audio, we only utilize the midi data for this experiment, for the audio part is unecessary for the classification of symbolic music.   

For the usage, please refer to the "Download" section of the official website.

[maestro_link]: https://magenta.tensorflow.org/datasets/maestro    

## Code Explanation
#### Preprocess
> Downloaded MAESTRO Midi dataset was preprocessed using [music21][music21_link], a toolkit for computer-aided musicology distributed by MIT. Preprocessing takes the following steps:   
> 1. Remove composers with too small number of data.
> 2. Extract notes from each instrument track
> 3. Divide into 0.05 second units
> 4. Mark note information on 3d matrix

[music21_link]: http://web.mit.edu/music21/

#### input
> Generated input takes the form of (129, 400, 128), where:   
> * 2 channel = onset + note
>   * channel[0] (onset) = binary
>   * channel[1] (note) = 0-128 velocity
> * 400 (x-dim) = time (0.05 sec)
> * 128 (y-dim) = 0-127 pitch 
<img src="https://user-images.githubusercontent.com/56469754/91077437-06a48580-e67c-11ea-9769-a5c19470a52e.png" width="40%" height="30%" title="input"></img><br/>

#### Model
> This dataset was trained using **ResNet50**

#### Adversarial Attack


#### Adversarial Training


## Similar Works

## Contact Information
* hylee817@yonsei.ac.kr
* ryan0507@yonsei.ac.kr
* hahala25@yonsei.ac.kr
