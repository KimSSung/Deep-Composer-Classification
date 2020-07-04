# Midi Genre Classification & Adversarial Attack
Genre classification using music in symbolic format (MIDI) implemented in PyTorch. 

## Abstract


## Requirements
#### Library
> * Python 3.7
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
> 3. Divide into 0.5 second units
> 4. Mark note information on 3d matrix

[music21_link]: http://web.mit.edu/music21/

#### input
> Generated input takes the form of (129, 400, 128), where:   
> * 129 channel = 128 instruments + 1 None
> * 400 (x-dim) = time (0.5 sec)
> * 128 (y-dim) = 0-127 pitch
> * each cell value = velocity
> <img src="https://user-images.githubusercontent.com/56469754/86505898-feab3080-be04-11ea-8ae6-90d8623352b4.jpg" width="40%" height="30%" title="input"></img><br/>

#### Model
This dataset was trained using **ResNet50**

![model](https://user-images.githubusercontent.com/56469754/86505899-01a62100-be05-11ea-81bb-174b37f66344.jpg)   

