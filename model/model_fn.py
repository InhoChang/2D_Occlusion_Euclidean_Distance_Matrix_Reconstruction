"""Define the model."""

import tensorflow as tf


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['edm']
    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, [-1, 16, 16, 1])


    assert images.get_shape().as_list() == [None, 16,16,1]

    out = images
    bn_momentum = params.bn_momentum

    with tf.variable_scope('conv1'):
        out = tf.layers.conv2d(out, 64, [7, 7], strides=1, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2) # (?, 8, 8, 64)
        out = tf.layers.dropout(out, 0.5, training=is_training)

    with tf.variable_scope('conv2'):
        out = tf.layers.conv2d(out, 64, [7, 7], strides=1, padding='same')
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2) # (?, 4, 4, 64)
        out = tf.layers.dropout(out, 0.5, training=is_training)


    assert out.get_shape().as_list() == [None, 4, 4, 64]

    with tf.variable_scope('deconv1'):
        out = tf.layers.conv2d_transpose(out, 64, [7, 7], strides=2, padding='same') # [H,W,output_k, Input_k] // (?, 8, 8, 64)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.dropout(out, 0.5, training=is_training)


    with tf.variable_scope('deconv2'):
        out = tf.layers.conv2d_transpose(out, 64, [7, 7], strides=2, padding='same') # (?, 16, 16, 64)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        logits = tf.layers.conv2d(out, 1, [1, 1], padding='same')

    return logits




def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['gt']
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [-1, 16, 16, 1])


    # labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions =logits

    # Define L2- loss and accuracy
    loss = tf.losses.mean_squared_error(labels = labels, predictions=predictions)
    # loss = tf.reduce_sum(tf.square(predictions - labels))
    # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
    # Define training step that minimizes the loss with the Adam optimizer

    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.image('logits', logits, max_outputs = 5)
    tf.summary.image('labels', labels, max_outputs = 5)





    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    # model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()


    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
