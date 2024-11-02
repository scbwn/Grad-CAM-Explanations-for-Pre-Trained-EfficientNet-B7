import os
import numpy as np
import tensorflow
import matplotlib as mpl
import matplotlib.pyplot as plt

from gradcam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam

create_model = tensorflow.keras.applications.EfficientNetB7
img_size = (600, 600)
preprocess_input = tensorflow.keras.applications.efficientnet.preprocess_input
decode_predictions = tensorflow.keras.applications.efficientnet.decode_predictions

last_conv_layer_name = "block7d_project_conv"


img_path = "./dog-and-cat-cover.jpg"

# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Make model
model = create_model(weights="imagenet")

# Remove last layer's softmax
model.layers[-1].activation = None

# Print model architecture
print(model.summary())

# Prepare image
img_array = preprocess_input(get_img_array(img_path, size=img_size))

# Print two top predicted classes
preds = model.predict(img_array)
print("Predicted:", decode_predictions(preds, top=2)[0])

# Class index of 'tabby cat' is 281 and of 'German Shephard' is 235
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=281)

# Save heatmap-overlayed image
plt.matshow(heatmap)
plt.show()
save_and_display_gradcam(img_path, heatmap)