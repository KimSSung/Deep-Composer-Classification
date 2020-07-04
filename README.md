# Midi Genre Classification & Adversarial Attack
Genre classification using music in symbolic format (MIDI) implemented in PyTorch. 

## Abstract


## Requirements
### Library
* Python 3.7
* PyTorch
* numpy
* matplotlib
* tqdm

### Dataset
Midi dataset was collected from two different websites cited as below:
* https://freemidi.org/   
Classical, Rock, Country genre midi files were collected from here and 300 of each were actually preprocessed and used to train the model.

- https://www.vgmusic.com/music/console/nintendo/gameboy   
GameMusic genre midi files were collected from here and 300 were actually used.


## Models
This dataset was trained using **ResNet50**

![model](https://user-images.githubusercontent.com/56469754/86505899-01a62100-be05-11ea-81bb-174b37f66344.jpg)   

### input
<img src="https://user-images.githubusercontent.com/56469754/86505898-feab3080-be04-11ea-8ae6-90d8623352b4.jpg" width="40%" height="30%" title="input"></img><br/>
