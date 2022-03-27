import tensorflow as tf

IMAGE_SIZE = (224, 340)
CROP_SIZE = (224, 224)
BATCH_SIZE = 128

def dataset(ds_path, *,
            train,
            preprocess_fn=None,
            image_size=IMAGE_SIZE,
            crop_size=CROP_SIZE,
            batch_size=BATCH_SIZE):
    """Returns a tf.data.Dataset pipeline suitable for training / inference.
    Training pipeline: uses data augmentation, random crops.
    Inference (test, val) pipeline: uses only central crop.
    
    Preprocessing function is applied at the end of each pipeline.
    """
    
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        ds_path, shuffle=train, label_mode='categorical',
        batch_size=batch_size, image_size=image_size)
    
    gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        zoom_range=0.2)
    
    @tf.function
    def augment(images, labels):
        aug_images = tf.map_fn(lambda image: tf.numpy_function(gen.random_transform,
                                                               [image],
                                                               tf.float32), 
                               images)
        aug_images = tf.ensure_shape(aug_images, images.shape)
        return aug_images, labels
    
    crop_layer = tf.keras.layers.experimental.preprocessing.RandomCrop(*crop_size)
    @tf.function
    def crop(images, labels):
        cropped_images = crop_layer(images, training=train)
        return cropped_images, labels
    
    if train:
        ds = ds.map(augment, tf.data.experimental.AUTOTUNE)
    ds = ds.map(crop, tf.data.experimental.AUTOTUNE)
    if preprocess_fn:
        ds = ds.map(preprocess_fn)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def model_probas(test_dataset, model):
    """Returns the predicted probabilities and the true labels
    for a given (inference) dataset on a given model."""
    y_test, y_probas = [], []
    
    for image_batch, label_batch in test_dataset:
        y_test.append(label_batch)
        y_probas.append(model.predict(image_batch))
    
    y_test, y_probas = (
        tf.concat(y_test, axis=0),
        tf.concat(y_probas, axis=0))
    
    return {
        'y_test': y_test,
        'y_probas': y_probas
    }


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate, BatchNormalization
from tensorflow.keras.layers import Dropout, GlobalAveragePooling2D, Dense, Flatten, Activation


def vgginnet_builder():
    base_model = VGG16(include_top=False, input_shape=(224, 224, 3))

    layer_name = 'block4_pool'
    feature_ex_model = Model(inputs=base_model.input, 
                             outputs=base_model.get_layer(layer_name).output, 
                             name='vgg16_features')
    feature_ex_model.trainable = False

    p1_layer = Lambda(vgg_preprocess, name='VGG_Preprocess')
    image_input = Input((224, 224, 3), name='Image_Input')
    p1_tensor = p1_layer(image_input)

    out =feature_ex_model(p1_tensor)
    feature_ex_model = Model(inputs=image_input, outputs=out)

    def naive_inception_module(layer_in, f1, f2, f3):
        # 1x1 conv
        conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)
        # 3x3 conv
        conv3 = Conv2D(f2, (3,3), padding='same', activation='relu')(layer_in)
        # 5x5 conv
        conv5 = Conv2D(f3, (5,5), padding='same', activation='relu')(layer_in)
        # 3x3 max pooling
        pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)
        # concatenate filters, assumes filters/channels last
        layer_out = Concatenate()([conv1, conv3, conv5, pool])
        return layer_out

    out = naive_inception_module(feature_ex_model.output, 64, 128, 32)
    num_classes = 2

    bn1 = BatchNormalization(name='BN')(out)
    f = Flatten()(bn1)
    dropout = Dropout(0.4, name='Dropout')(f)
    desne = Dense(num_classes, activation='softmax', name='Predictions')(dropout)

    model = Model(inputs=feature_ex_model.input, outputs=desne)
    return model
