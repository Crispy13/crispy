from tensorflow import keras
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras import backend as K

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    """
    Makes grad cam heatmap.
    
    Parameters
    ----------
    img_array : an input image
    
    model : a keras model object
    
    last_conv_layer_name : a string
        The gradients of y^c w.r.t the layer activation will be calculated.
    
    classifier_layer_names : a sequence
        layer names from next to the last conv layer to the output layer.
    
    """
    
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        if layer_name == 'predictions':
            l = keras.layers.Dense(model.get_layer(layer_name).units, name = "predictions_gradcam")
            x = l(x)
            l.set_weights(model.get_layer(layer_name).get_weights())
        else:
            x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap


###
def build_guided_model(model):
    """
    Builds guided model 
    
    """
    
    model_copied = keras.models.clone_model(model)
    model_copied.set_weights(model.get_weights())
    
    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy>0, tf.float32) * tf.cast(x>0, tf.float32) * dy
        return tf.nn.relu(x), grad

    layer_dict = [layer for layer in model_copied.layers[1:] if hasattr(layer,'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu
    
    return model_copied

###
def guided_backprop(model, img, layer_name):
    """
    Returns guided backpropagation image.
    
    Parameters
    ----------
    model : a keras model object
    
    img : an img to inspect with guided backprop.
    
    layer_name : a string
        a layer name for calculating gradients.
    
    
    """
    part_model = keras.Model(model.inputs, model.get_layer(layer_name).output)
    
    with tf.GradientTape() as tape:
#         model_input = indentity_model(img)
#         tape.watch(model_input)
#         part_output = part_model(img)
        f32_img = tf.cast(img, tf.float32)
        tape.watch(f32_img)
        part_output = part_model(f32_img)
        
    grads = tape.gradient(part_output, f32_img)[0].numpy()
    
    # delete copied model
    del part_model
    
    return grads


###
def guided_grad_cam(guided_backprop, gradcam, target_size = (224, 224)):
    """
    Make guided grad cam image.
    
    Parameters
    ----------
    guided_backprop : a numpy array 
        guided_backpropagation graidents array
        
    gradcam : a numpy array
        gradcam heatmap before being resized.
    
    target_size : a tuple
        resizing target size.
    
    
    
    """
    resized_gradcam = cv2.resize(gradcam, guided_backprop.shape[:-1])
    gradcam_r = np.repeat(resized_gradcam[...,None], 3, axis = 2)
    
    return guided_backprop * gradcam_r


###
def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    
    return x
