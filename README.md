# Deep Composer Classification Using Symbolic Representation

## Abstract
This research has its purpose in accelerating the usage of symbolic representation of music through showing its potential in improving accuracy and efficiency in recognizing and handling musical information.
Classification task, whether it's concerning genre, style, era, or composer as each category, is a relatively common in the music domain. To the best of our knowledge, all the previous works have been mostly attempted using audio/wav features, such as MFCC, mel-spectrogram. However, using only audio/wav format to detect, analyze and maintain music has its limitations. Through this research, we show that using symbolic representation(Midi, music piece, etc) can strengthen certain aspects of music detection and classification, which is often the basis of many tasks in music industry, and suggest symbolic representation as a possible solution to the current limitations and problems present in working with audio/wav format. Furthermore, we went ahead and tested serveral adversarial attack techniques on this format of music. 

## Requirements
#### Library
> * Python >= 3.6.9
> * PyTorch = 1.4.0
> * torchvision = 
> * py_midicsv = 1.10.0
> * numpy = 1.18.1
> * sklearn = 
> * matplotlib = 3.1.3
> * music21 = 5.7.2
> * tqdm

## How to Run
### Generate
### Split

### Train

#### basetrain
        python main.py --gpu [gpu to use]
                       --mode basetrain
                       --model_name [model to use]
                       --epochs [epoch #]
                       --optim [optimizer to use]
                       --transform [Transpose / Tempo]
                       --load_path ['/PATH/TO/TRAIN.TXT_AND_VALID.TXT/']
                       --save_path ['/PATH/TO/SAVE/MODEL_AND_LOADER/']
#### advtrain
        python main.py --gpu [gpu to use]
                       --mode advtrain
                       --model_name [model to use]
                       --epochs [epoch #]
                       --optim [optimizer to use]
                       --transform [Transpose / Tempo]
                       --input_path ['/PATH/TO/ATTACKED/INPUT/']
                       --save_path ['/PATH/TO/SAVE_MODEL_AND_LOADER/']
### Attack
        python main.py --gpu [gpu to use]
                       --mode attack 
                       --load_path [/PATH/TO/SAVED_LOADER/] 
                       --save_path [/PATH/TO/SAVE/ATTACK_EXAMPLES/] 
                       --epsilons ['ep0, ep1, ep2, ... epn']
                       --save_atk [True/False]

### Convert

## How to monitor
        tensorboard --logdir=trainlog

## Actual Examples
        python main.py --gpu 0
                       --mode basetrain
                       --model_name resnet50
                       --epochs 100 --optim SGD
                       --load_path '/data/split/'
                       --save_path '/data/drum/dataset/'
                       
        python main.py --gpu 3
                       --mode attack 
                       --load_path '/data/drum/bestmodel/' 
                       --save_path '/data/attacks/' 
                       --epsilons '0.05, 0.1, 0.2, 0.4, 0.6' 
                       --save_atk True

## Dataset
[MAESTRO][maestro_link]: (MIDI and Audio Edited for Synchronous TRacks and Organization) is a dataset composed of over 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.     

Specifically, we used **v2.0.0** version of the dataset. Although the big advantage of using this dataset is the fine alignment between midi & audio, we only utilize the midi data for this experiment, for the audio part is unecessary for the classification of symbolic music.   

For the usage, please refer to the "Download" section of the official website.

[maestro_link]: https://magenta.tensorflow.org/datasets/maestro    

### Preprocess
> Downloaded MAESTRO Midi dataset was preprocessed using [music21][music21_link], a toolkit for computer-aided musicology distributed by MIT. Preprocessing takes the following steps:           
> **1. Remove composers with too small number of data.**           
> **2. Extract notes from each track**        
> **3. Divide into 0.05 second units**         
> **4. Mark note information on 3d matrix**        

> Uneven Distribution of data       
> <img src="https://user-images.githubusercontent.com/56469754/91077423-00160e00-e67c-11ea-8977-01366e0ad5e7.png" width="80%" height="30%" title="duration"></img><br/>  
> Remaining 14 composers after step(1)    
> <img src="https://user-images.githubusercontent.com/56469754/91096404-f18a1f80-e698-11ea-8878-a54f3128b384.png" width="30%" height="20%" title="duration"></img><br/> 


[music21_link]: http://web.mit.edu/music21/

#### input
> Generated input takes the form of (2, 400, 128), where:   
> * 2 channel = onset + note
>   * channel[0] (onset) = binary
>   * channel[1] (note) = 0-128 velocity
> * 400 (x-dim) = time (0.05 sec)
> * 128 (y-dim) = 0-127 pitch     

<img src="https://user-images.githubusercontent.com/56469754/91077437-06a48580-e67c-11ea-9769-a5c19470a52e.png" width="40%" height="30%" title="input"></img><br/>

#### Model
> This dataset was experimented on different model configurations:             

|  <center> Model </center> |  <center> Train Acc </center> |  <center> Valid Acc</center> |         
|:--------:|--------:|--------:|         
|**Resnet** | <center> % </center> | <center> % </center> |        
|**Resnet (7,3)** | <center> % </center> | <center> % </center> |        
|**Wide Resnet** | <center> % </center> | <center> % </center> |         


#### Adversarial Attack


#### Adversarial Training


## Similar Works

## Contact Information
* hylee817@yonsei.ac.kr
* ryan0507@yonsei.ac.kr
* hahala25@yonsei.ac.kr


