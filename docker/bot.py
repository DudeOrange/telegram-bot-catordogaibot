from PIL import Image
from urllib.request import urlretrieve
from torchvision import models, transforms
import torch
import torch.nn as nn
import numpy as np

import telebot

token = 'your_token'
bot = telebot.TeleBot(token)


model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('resnet18-fulldata-model.pth', map_location='cpu'))
model.eval()

transform=transforms.Compose([
                           transforms.Resize((224, 224)),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                         
                       ])

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.send_message(message.chat.id, 'Send me a picture with cat or dog and I will say dog or cat on it')

@bot.message_handler(content_types=['photo'])
def repeat_all_messages(message):
    #get a photo from a message
    file_id = message.photo[-1].file_id
    newFile = bot.get_file(file_id)
    file_name = newFile.file_path[7:]
    path = "https://api.telegram.org/file/bot" + token + "/" + newFile.file_path
    urlretrieve(path, file_name)
    #preparing photo to the data model
    test_filename = file_name
    test_image = Image.open(test_filename)
    test_image_tensor = transform(test_image).float()
    test_image_tensor = test_image_tensor.unsqueeze_(0)
    #get the result from the model
    output = model(test_image_tensor)
    output = output.detach().numpy()
    out = np.argmax(output)
    if out == 0:
        bot.send_message(message.chat.id, 'This is a cat')
    else:
        bot.send_message(message.chat.id, 'This is a dog')    

if __name__ == '__main__':
     bot.polling(none_stop=True)