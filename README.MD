# Telegram bot "catordogaibot" #
---
## About pet-project ##
In this project, I wanted to go through the whole cycle of implementation deep learning model. So I chose classic Image Classification DL problem: define cat or dog on the picture.

Choice dataset and tools for project:

- I use CC0-licensed "Cat and Dog" dataset from Kaggle. You can download it [here](https://www.kaggle.com/tongpython/cat-and-dog).
- I use pretrained resnet18 from Pytorch library as the DL model.
- For the applying model, I wrote telegram bot. Bots are the easiest way if you don't want to create your own site or application. You can find my bot on the Telegram by the name "@catordogaibot".
- I chose [Heroku](https://www.heroku.com/home#) for hosting my bot. Actually bot can work on your own local machine, so I also added docker image to this repo.

## How to use it ##
1. If you want to improve or change the neural network, 
you have to make sure that you have Anaconda. You can download Anaconda [here](https://www.anaconda.com/products/individual). Also you will need Pytorch library. [Here](https://pytorch.org/get-started/locally/) installation instructions.
2. If you want to deploy the bot on the server, I will look how on Heroku example.

	First of all, you will need to create account on [Heroku](https://signup.heroku.com/). And you have to make sure that you have [Git](https://git-scm.com/downloads). Then execute commands in command line:
    
		cd YourWorkDirectory
		heroku login
		heroku create #create app on server
		heroku git:remote -a YourAppName #get access to the your app repository
		git init
		git add .
		git commit -am "comment"
		git push heroku master
		heroku ps:scale worker=1 #start working app
3. If you want to run Docker container on your local machine, make sure that you install [Docker](https://docs.docker.com/get-docker/). Then execute commands in command line:

		cd YourWorkDirectory
		docker build -t simple-build . #create image
		docker run -p 8888:8888 NameYourImage #run your image
		#That's all. You can check that all works with command:
		docker ps -s

P.S. Don't forget to change the paths to files and the bot token to your own.