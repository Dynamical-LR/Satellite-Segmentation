<div style="text-align:center; margin-bottom: 30px">
  <h2>Satellite-Segmentation Network</h2>
</div>

[comment]: <> ("Docs Badges goes there")

<div class="container badges" 
style="display: flex; justify-content: center; column-gap: 5px; margin-bottom: 30px">

<a href="https://github.com/LovePelmeni/Credit-Card-Approval-Project/pulse" alt="Activity">
        <img src="https://img.shields.io/badge/version-1.2.3-blue" /></a>

<a href="https://github.com/LovePelmeni/Credit-Card-Approval-Project/pulse" alt="Activity">
        <img src="https://img.shields.io/github/commit-activity/m/badges/shields" /></a>
    
<a href="https://circleci.com/gh/badges/shields/tree/master">
    <img src="https://img.shields.io/circleci/project/github/badges/shields/master" alt="build status">
</a>
    
<a href="https://circleci.com/gh/badges/daily-tests">
    <img src="https://img.shields.io/circleci/project/github/badges/daily-tests?label=service%20tests" alt="service-test status">
</a>

<a href="https://coveralls.io/github/badges/shields">
    <img src="https://img.shields.io/coveralls/github/badges/shields"
            alt="coverage">
</a>

</div>

<div style="margin-bottom: 40px">

This is home to `Satellite-Segmentation-Network`, an offline neural network
for segmenting satellite images
</div> 

## Technologies & core libraries

1. *Python* (Main programming language used for development)
2. *Pytorch* (framework for DL Development)
4. *PIL* - for efficient image management
5. *opencv* - for efficient image processing
6. *pandas and numpy* - for working with tabular data

## Versioning

1. `3.9` <= Python <= `3.10`
2. Pytorch >= `20.10.12`
3. PIL >= `10.1.0`
3. opencv >= `4.8.1.78`


## Directory overview 

`notebooks` - contains all source code for training pipeline and data generation

`dataset` - contains dataset abstract class for working with the image data

`model` - directory for storing trained network

`network_trainer` - contains module for efficient training of the network,
provides plenty of options for customizing loss, scheduler, optimizer, etc...

`losses` - contains implementation of loss functions, used in the project

`proj_requirements` - contains project dependencies instructions 


# Usage

### Clone project using following command

```
$ git clone https://github.com/Dynamical-LR/Satellite-Segmentation
```

## Local Development
### Setting up Python Virtual Environment

```
# Creating new virtual environment

$ python3 -m venv fn_env
$ source ./fn_env/bin/activate 
```
### Installing project dependencies

```
$ pip install -r proj_requirements/module_requirements.txt
```
### Enjoy working with your model
