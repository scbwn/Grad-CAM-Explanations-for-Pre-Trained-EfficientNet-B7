import os
import numpy as np
import tensorflow

def get_img_array(img_path, size):
    img = tensorflow.keras.utils.load_img(img_path, target_size=size)
    array = tensorflow.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create model
    grad_model = tensorflow.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of the top predicted class for our input image 
    # with respect to the activations of the last conv layer
    with tensorflow.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tensorflow.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tensorflow.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[:, tensorflow.newaxis]
    heatmap = tensorflow.squeeze(heatmap)


    heatmap = tensorflow.maximum(heatmap, 0) / tensorflow.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = tensorflow.keras.utils.load_img(img_path)
    img = tensorflow.keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    
    cmap = mpl.colormaps["jet"]
    cmap_colors = cmap(np.arange(256))[:, :3]
    c_heatmap = cmap_colors[heatmap]
    c_heatmap = tensorflow.keras.utils.array_to_img(c_heatmap)
    c_heatmap = c_heatmap.resize((img.shape[1], img.shape[0]))
    c_heatmap = tensorflow.keras.utils.img_to_array(c_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = c_heatmap * alpha + img
    superimposed_img = tensorflow.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)