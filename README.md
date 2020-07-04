# Midi Genre Classification & Adversarial Attack
Genre classification using music in symbolic representation (MIDI) implemented in PyTorch. 

## Abstract
This research has its purpose in accelerating the usage of symbolic representation of music through showing its potential in improving accuracy and efficiency in recognizing and handling musical information.
Genre classification in the music domain is a relatively common problem. To the best of our knowledge, all the previous works including the ones that show SOTA performance have been attempted using audio/wav features, such as MFCC, mel-spectrogram. However, using only audio/wav format to detect, analyze and maintain music has its limitations. Through this research, we show that using symbolic representation(Midi, music piece, etc) can strengthen certain aspects of music detection and classification, which is often the basis of many tasks in music industry, and suggest symbolic representation as a possible solution to the current limitations and problems present in working with audio/wav format. Furthermore, we went ahead and tested serveral adversarial attack techniques on this format of music. 

## Requirements
#### Library
> * Python >= 3.6.9
> * PyTorch
> * numpy
> * matplotlib
> * music21
> * tqdm

#### Dataset
> Midi dataset was collected from two different websites cited as below:
> * https://freemidi.org/   
> Classical, Rock, Country genre midi files were collected from here and 300 of each were actually preprocessed and used to train the model.
> 
> - https://www.vgmusic.com/music/console/nintendo/gameboy   
> GameMusic genre midi files were collected from here and 300 were actually used.

## Code Explanation
#### Preprocess
> Collected Midi dataset was preprocessed using [music21][music21_link], a toolkit for computer-aided musicology distributed by MIT. Preprocessing takes the following steps:   
> 1. Remove drum track (optional)
> 2. Extract notes from each instrument track
> 3. Divide into 0.05 second units
> 4. Mark note information on 3d matrix

[music21_link]: http://web.mit.edu/music21/

#### input
> Generated input takes the form of (129, 400, 128), where:   
> * 129 channel = 128 instruments + 1 None
> * 400 (x-dim) = time (0.05 sec)
> * 128 (y-dim) = 0-127 pitch
> * each cell value = velocity   
<img src="https://user-images.githubusercontent.com/56469754/86505898-feab3080-be04-11ea-8ae6-90d8623352b4.jpg" width="40%" height="30%" title="input"></img><br/>

#### Model
> This dataset was trained using **ResNet50**
> <img src="https://user-images.githubusercontent.com/56469754/86505899-01a62100-be05-11ea-81bb-174b37f66344.jpg" width="80%" height="50%" title="model"></img><br/>

#### Adversarial Attack


#### Adversarial Training


## Similar Works

## Contact Information
* hylee817@yonsei.ac.kr
* ryan0507@yonsei.ac.kr
* hahala25@yonsei.ac.kr
