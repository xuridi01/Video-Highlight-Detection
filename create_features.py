import tensorflow as tf
import i3d
import numpy as np

_IMAGE_SIZE = 224
_FRAMES_PER_SEGMENT = 50 
# Kinetics-400 classes
_NUM_CLASSES = 400  

# Paths to I3D pre-trained checkpoints
_CHECKPOINT_PATHS = {
    'RGB': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'Flow': 'data/checkpoints/flow_imagenet/model.ckpt'
}

def extract_i3d_features(frames, model_type):
    tf.reset_default_graph()

    input_shape = (None, _FRAMES_PER_SEGMENT, _IMAGE_SIZE, _IMAGE_SIZE, 3) if model_type == 'RGB' else (None, _FRAMES_PER_SEGMENT, _IMAGE_SIZE, _IMAGE_SIZE, 2)
    input_tensor = tf.placeholder(tf.float32, shape=input_shape)

    # Inference
    with tf.variable_scope(model_type):
        model = i3d.InceptionI3d(_NUM_CLASSES, spatial_squeeze=True, final_endpoint='Mixed_5c')
        features, _ = model(input_tensor, is_training=False, dropout_keep_prob=1.0)
        features = tf.reduce_mean(features, axis=[2, 3])
        features = tf.reduce_mean(features, axis=1)

    # Load pre-trained weights
    variable_map = {var.name.replace(':0', ''): var for var in tf.global_variables() if model_type in var.name}
    saver = tf.train.Saver(var_list=variable_map, reshape=True)

    with tf.Session() as sess:
        saver.restore(sess, _CHECKPOINT_PATHS[model_type])
        print(f"{model_type.upper()} I3D model loaded.")

        # Extract features
        feed_dict = {input_tensor: frames}
        extracted_features = sess.run(features, feed_dict=feed_dict)

    print(extracted_features.shape)
    return extracted_features

def load_and_save(rgb, opt, rgb_fetures, opt_features):
    data_rgb = np.load(rgb)
    features = extract_i3d_features(data_rgb, 'RGB')
    np.save(rgb_fetures, features)


    data_opt = np.load(opt)
    features = extract_i3d_features(data_opt, model_type='Flow')
    np.save(opt_features, features)