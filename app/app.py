from glob import glob

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

ResNet50_model = tf.keras.applications.resnet50.ResNet50(weights="imagenet")

dog_names = [item[20:-1] for item in sorted(glob("data/dog_images/train/*/"))]


def face_detector(img_path):
    """
    Detect if a human face is present in the given image.

    Args:
        img_path (str): The file path to the image.

    Returns:
        bool: True if one or more faces are detected, False otherwise.
    """
    face_cascade = cv2.CascadeClassifier(
        "../data/haarcascades/haarcascade_frontalface_alt.xml"
    )
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def ResNet50_predict_labels(img_path):
    """
    Predict the label of an image using the pre-trained ResNet50 model.

    This function processes an image located at `img_path` using ResNet50's
    preprocessing function, converts it to a tensor, and returns the predicted
    label as the index of the highest value in the prediction vector.

    Args:
        img_path (str): The file path to the image.

    Returns:
        int: The index of the predicted label.
    """

    img = tf.keras.applications.resnet50.preprocess_input(
        path_to_tensor(img_path)
    )
    return np.argmax(ResNet50_model.predict(img, verbose=0))


def dog_detector(img_path):
    """
    Determine whether a dog is detected in an image.

    This function uses the ResNet50 model to predict labels for the image.
    If the prediction falls within the range of dog breed labels (151 to 268),
    the function returns True, indicating a dog has been detected. Otherwise,
    it returns False.

    Args:
        img_path (str): The file path to the image.

    Returns:
        bool: True if a dog is detected, False otherwise.
    """
    prediction = ResNet50_predict_labels(img_path)
    return (prediction <= 268) & (prediction >= 151)


def path_to_tensor(img_path):
    """
    Convert an image file into a normalized 4D tensor.

    This function loads an image from the given file path, resizes it to the
    target size of (224, 224), converts the image to a tensor format, and
    normalizes it. The resulting tensor is expanded into a 4D array (batch size,
    height, width, channels). If the image file is corrupted or unreadable, a
    warning message is printed, and None is returned.

    Args:
        img_path (str): The file path to the image.

    Returns:
        np.ndarray or None: The 4D tensor representing the image, or None if the
        image is corrupted or unreadable.
    """
    try:
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(224, 224)
        )
        x = tf.keras.preprocessing.image.img_to_array(img)
        # Normalize the image tensor
        return np.expand_dims(x, axis=0)
    except IOError:
        print(f"Warning: Skipping corrupted image {img_path}")
        return None


def create_model():
    Xception_model = tf.keras.models.Sequential()
    Xception_model.add(
        tf.keras.layers.GlobalAveragePooling2D(input_shape=(7, 7, 2048))
    )
    Xception_model.add(tf.keras.layers.Dense(1024, activation="relu"))
    Xception_model.add(tf.keras.layers.Dense(133, activation="softmax"))

    return Xception_model


def extract_Xception(tensor):
    return tf.keras.applications.xception.Xception(
        weights="imagenet", include_top=False
    ).predict(tf.keras.applications.xception.preprocess_input(tensor))


def Xception_predict_breed(img_path, model, names):
    """
    Predicts the dog breed from an image using a pre-trained Xception model.

    This function extracts bottleneck features from the input image using a
    pre-trained Xception model, then uses these features to predict
    the dog breed.

    Args:
        img_path (str): Path to the image file.

    Returns:
        str: The predicted dog breed from the model.
    """
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature, verbose=0)
    # return dog breed that is predicted by the model
    return names[np.argmax(predicted_vector)]


def predict_breed(img_path, model, names):
    """
    Predicts whether the image is a dog, human, or neither, and predicts the
    corresponding dog breed if applicable.

    This function first displays the image, then checks if the image contains
    a dog or a human face. If it's a dog, it predicts the dog's breed. If it's
    a human, it suggests what dog breed the person might resemble. If neither
    a dog nor a human is detected, it returns a message indicating that.

    Args:
        img_path (str): Path to the image file.

    Returns:
        None: Outputs a message indicating the detection result and predicted
        dog breed.
    """
    predict_img = Xception_predict_breed(img_path, model, names)

    if dog_detector(img_path):
        title = f'This is a Dog with a chance of being of the breed: {predict_img.split(".", 1)[1]}.'
        return title

    if face_detector(img_path):
        title = f'This is a Human, but it could be a dog breed: {predict_img.split(".", 1)[1]}.'
        return title

    else:
        title = "This is neither dog nor human."
        return title


checkpoint_Xception_filepath = "saved_models/checkpoint.model.Xception.keras"
Xception_model = create_model()
Xception_model.load_weights(checkpoint_Xception_filepath)


st.set_page_config(
    page_title="Dog Breed Classifier",
)


# Sidebar
with st.sidebar:
    st.subheader("Data Scientist Nanodegree")
    st.caption("=== Fernanda Rodriguez ===")

    st.subheader(":arrow_up: Upload image")
    uploaded_file = st.file_uploader("Choose image")


# Body
st.header("Dog Breed Classifier")

if uploaded_file is not None:
    st.subheader(":camera: Input")
    st.image(uploaded_file)
    if st.button(":arrows_counterclockwise: Predict Dog Breed"):
        with st.spinner("Wait for it..."):
            prediction = predict_breed(
                uploaded_file, Xception_model, dog_names
            )

            st.caption(":white_check_mark: Prediction:")
            st.subheader(f"{prediction}")
