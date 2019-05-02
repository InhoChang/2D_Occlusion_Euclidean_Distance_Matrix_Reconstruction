"""Create the input data pipeline using `tf.data`"""

import tensorflow as tf

def _parse_function(filename, label, size):
    image_string = tf.read_file(filename)
    label_string = tf.read_file(label)

    image_decoded = tf.image.decode_png(image_string, channels=1) # int value
    label_decoded = tf.image.decode_png(label_string, channels=1) # int value

    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    lb = tf.image.convert_image_dtype(label_decoded, tf.float32)

    resized_image = tf.image.resize_images(image, [size, size]) # float32 value between 0~1
    resized_label = tf.image.resize_images(lb, [size, size]) # float32 value between 0 ~1

    return resized_image, resized_label

## test
# filename = train_filenames[0]
# label = train_labels[0]

# sess = tf.InteractiveSession()



def input_fn(is_training, input, gt, params):
    num_samples = len(input)
    assert len(input) == len(gt)

    parse_fn = lambda f, l: _parse_function(f, l, params.image_size)
    # https: // stackoverflow.com / questions / 48889482 / feeding - npy - numpy - files - into - tensorflow - data - pipeline
    if is_training:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(input), tf.constant(gt)))
        # dataset = (tf.data.Dataset.from_tensor_slices((input, gt))
                   .shuffle(num_samples)
                   .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
                   .batch(params.batch_size)
                   .prefetch(1)  # make sure you always have one batch ready to serve
                   )

    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(input), tf.constant(gt)))
        # dataset = (tf.data.Dataset.from_tensor_slices((input, gt))
                   .map(parse_fn)
                   .batch(params.batch_size)
                   .prefetch(1)  # make sure you always have one batch ready to serve
                   )


    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()  # create the iterators from the dataset
    edm, gt = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {'edm': edm, 'gt': gt, 'iterator_init_op': iterator_init_op}

    return inputs
