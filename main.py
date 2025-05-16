import telebot
from bot_logic import gen_pass, gen_emoji, flip_coin
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

bot = telebot.TeleBot("your bots token")

def start_model(model_path):
  # Disable scientific notation for clarity
  np.set_printoptions(suppress=True)

  # Load the model
  model = load_model(model_path, compile=False)
  
  return model

def predict_image(image_path, labels_path, model):
    # Load the labels
  class_names = open(labels_path, "r", encoding="UTF-8").readlines()

  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  image = Image.open(image_path).convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array

  # Predicts the model
  prediction = model.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]

  return class_name[2: -1]

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hi! I'm your new telegram bot. Use commands like /pass, /emodji или /coin. Also you can upload a photo here and I'm gonna analyze whats on that photo.")

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    # Checking if message contains a photo
    if not message.photo:
        return bot.send_message(message.chat.id, "You forgot to upload a photo :(")

    # Getting and saving the file
    file_info = bot.get_file(message.photo[-1].file_id)
    file_name = file_info.file_path.split('/')[-1]
    
    # Uploading and saving file
    downloaded_file = bot.download_file(file_info.file_path)
    file_path = "images/" + file_name # saving images to a specified path to organize photos
    with open(file_path, 'wb') as new_file:
        new_file.write(downloaded_file)
        
    bot.send_message(message.chat.id, "Starting up model..")    
    model = start_model("/model/your_keras_model.h5") # your model can be named anything just make sure that the name and file extension actually match your model
    bot.send_message(message.chat.id, "Model has been started, starting inference")
    label = predict_image(file_path, "/model/your_labels.txt", model) # labels for the classes that your model contain (change the name for the "your_labels.txt" to your label document file name)
    bot.send_message(message.chat.id, f"I think this is.. {label}")

@bot.message_handler(commands=['pass'])
def send_password(message):
    bot.reply_to(message, "How long do you want to have your password? For exmpl. 10")
    if message.text.isdigit():
        password_len = int(message.text)
        password = gen_pass(password_len)  # Getting the length of the password, for exmpl. 10 symbols
        bot.reply_to(message, f"Your generated password: {password}")
    else:
        return bot.send_message(message.chat.id, "You have to reply with numbers e.g. 10")

@bot.message_handler(commands=['emoji'])
def send_emoji(message):
    emoji = gen_emoji()
    bot.reply_to(message, f"Here's an emoji': {emoji}")

@bot.message_handler(commands=['coin'])
def send_coin(message):
    coin = flip_coin()
    bot.reply_to(message, f"It's: {coin}")

# Starting up a bot itself
bot.polling()