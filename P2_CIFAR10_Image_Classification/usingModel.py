import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def numToClass(index):
    classes = [
        "Airplanes",
        "Cars",
        "Birds",
        "Cats",
        "Deer",
        "Dogs",
        "Frogs",
        "Horses",
        "Ships",
        "Trucks"
    ]
    return classes[index] if 0 <= index < len(classes) else None

model_path = 'saved_models/keras_cifar10_trained_model_Augmentation.h5'
cnn_model = load_model(model_path)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(32, 32))
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_path):
    img_array = preprocess_image(image_path)
    prediction = cnn_model.predict_on_batch(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_name = numToClass(predicted_class)
    return class_name

image_paths = [
    '/home/solomons/MachineLearningStuff/P2_CIFAR10_Image_Classification/example_images/frog7.png', 
    '/home/solomons/MachineLearningStuff/P2_CIFAR10_Image_Classification/example_images/dog4.png'
]

for image_path in image_paths:
    class_name = classify_image(image_path)
    img = load_img(image_path, target_size=(32, 32))
    
    plt.figure()
    plt.imshow(img)
    plt.title(f'Predicted Class: {class_name}')
    plt.axis('off')
    
    output_path = image_path.replace('.jpg', '_predicted.jpg').replace('.png', '_predicted.png')
    plt.savefig(output_path)
    plt.close()
    print('OUTPUT : \n\n')
    print(f'Processed {image_path}\n, predicted class: {class_name}\n, saved as {output_path}\n')
