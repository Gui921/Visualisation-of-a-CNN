import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw


def visualize_filters(model, layer_name):
    layer = model.get_layer(name=layer_name)
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1
    plt.title("Filters")
    for i in range(n_filters):
        f = filters[:, :, :, i]
        ax = plt.subplot(2, 3, ix)  
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(f[:, :, 0], cmap='gray')  
        ix += 1

    plt.show()

def save_filters(model, layer_name):
    layer = model.get_layer(name=layer_name)
    filters, biases = layer.get_weights()
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 6, 1

    for i in range(n_filters):
        f = filters[:, :, :, i]
        plt.imshow(f[:, :, 0], cmap='gray')  
        if layer_name == 'conv2d_2':
            plt.savefig("filters/filter " + str(i) + "_2")
        else:
            plt.savefig("filters/filter " + str(i))

        plt.clf()


def visualize_activations(model, layer_name, img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_array)
    num_activations = activations.shape[-1]
    plt.figure(figsize=(20, 20))

    for i in range(num_activations):
        ax = plt.subplot(8, 4, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])

        if len(activations.shape) == 4:
            plt.imshow(activations[0, :, :, i], cmap='gray')
        else:
            plt.imshow(activations[0, :, i], cmap='gray')

    plt.show()

def save_activations(model, layer_name, img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    layer_outputs = [layer.output for layer in model.layers if layer.name == layer_name]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(img_array)
    
    plt.figure(figsize=(20, 20))

    for i in range(6):

        if len(activations.shape) == 4:
            plt.imshow(activations[0, :, :, i], cmap='gray')
        else:
            plt.imshow(activations[0, :, i], cmap='gray')
        
        if layer_name == 'conv2d_2':
            plt.savefig("fmaps/fmap " + str(i) + "_2")
        else:
            plt.savefig("fmaps/fmap " + str(i))

def run(model_type):
    global model
    if model_type == 1:
        model = tf.keras.models.load_model("models/1-layer.h5")
    else:
        model = tf.keras.models.load_model("models/2-layer.h5")

    image_path = 'pred.png' 
    img = Image.open(image_path).convert('L') 
    img = img.resize((28, 28)) 
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    pred_perc = np.max(prediction)
    return predicted_digit, pred_perc
