import tensorflow as tf
from tensorflow.contrib.losses.python.metric_learning.metric_loss_ops import pairwise_distance


def dist_weighted_sampling(labels, embeddings, high_var_threshold=0.5, nonzero_loss_threshold=1.4, neg_multiplier=1):
    """
    Distance weighted sampling.
    # References
        - [sampling matters in deep embedding learning]
          (https://arxiv.org/abs/1706.07567)

    # Arguments:
        labels: 1-D tf.int32 `Tensor` with shape [batch_size] of
          multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
          be l2 normalized.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        a_indices: indices of anchors.
        anchors: sampled anchor embeddings.
        positives: sampled positive embeddings.
        negatives: sampled negative embeddings.
    """
    if not isinstance(neg_multiplier, int):
        raise ValueError("`neg_multiplier` must be an integer.")
    n = tf.size(labels)
    if not isinstance(embeddings, tf.Tensor):
        embeddings = tf.convert_to_tensor(embeddings)
    d = embeddings.shape[1].value

    distances = pairwise_distance(embeddings, squared=False)
    # cut off to void high variance.
    distances = tf.maximum(distances, high_var_threshold)

    # subtract max(log(distance)) for stability
    log_weights = (2 - d) * tf.log(distances + 1e-16) - 0.5 * (d - 3) * tf.log(1 + 1e-16 - 0.25 * (distances**2))
    weights = tf.exp(log_weights - tf.reduce_max(log_weights))

    # sample only negative examples by setting weights of the same class examples to 0.
    lshape = tf.shape(labels)
    assert lshape.shape == 1
    labels = tf.reshape(labels, [lshape[0], 1])
    adjacency = tf.equal(labels, tf.transpose(labels))
    adjacency_not = tf.logical_not(adjacency)
    mask = tf.cast(adjacency_not, tf.float32)

    # number of negative/positive samples to sampling per sample.
    # For imbalanced data, this sampling method can be a sample weighted method.
    adjacency_ex = tf.cast(adjacency, tf.int32) - tf.diag(tf.ones(n, dtype=tf.int32))
    m = tf.reduce_sum(adjacency_ex, axis=1)
    if tf.reduce_min(m) == 0:
        m = tf.diag(tf.cast(tf.equal(m,0), tf.int32))
        adjacency_ex += m
    k = tf.maximum(tf.reduce_max(m),1) * neg_multiplier

    pos_weights = tf.cast(adjacency_ex, tf.float32)

    weights = weights * mask * tf.cast(distances < nonzero_loss_threshold, tf.float32)
    weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + 1e-16)

    #  anchors indices
    a_indices = tf.reshape(tf.range(n), (-1,1))
    a_indices = tf.tile(a_indices, [1, k])
    a_indices = tf.reshape(a_indices, (-1,))

    # negative sampling
    def neg_sampling(i):
        s = tf.squeeze(tf.multinomial(tf.log(tf.expand_dims(weights[i] + 1e-16, axis=0)), k, output_dtype=tf.int32), axis=0)
        return s

    n_indices = tf.map_fn(neg_sampling, tf.range(n), dtype=tf.int32)
    n_indices = tf.reshape(n_indices, (-1,))

    # postive samping
    def pos_sampling(i):
        s = tf.squeeze(tf.multinomial(tf.log(tf.expand_dims(pos_weights[i] + 1e-16, axis=0)), k, output_dtype=tf.int32), axis=0)
        return s

    p_indices = tf.map_fn(pos_sampling, tf.range(n), dtype=tf.int32)
    p_indices = tf.reshape(p_indices, (-1,))

    anchors = tf.gather(embeddings, a_indices, name='gather_anchors')
    positives = tf.gather(embeddings, p_indices, name='gather_pos')
    negatives = tf.gather(embeddings, n_indices, name='gather_neg')

    return a_indices, anchors, positives, negatives


def margin_based_loss(labels, embeddings, beta_in=1.0, margin=0.2, nu=0.0, high_var_threshold=0.5,
                      nonzero_loss_threshold=1.4, neg_multiplier=1):
    """
    Computes the margin base loss.
    # References
        - [sampling matters in deep embedding learning]
          (https://arxiv.org/abs/1706.07567)

    Args:
        labels: 1-D. tf.int32 `Tensor` with shape [batch_size] of multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        beta_in: float,int or 1-D, float `Tensor` with shape [labels_size] of multi-class boundary parameters.
        margin: Float, margin term in the loss function.
        nu: float. Regularization parameter for beta.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        margin_based_Loss: tf.float32 scalar
    """

    a_indices, anchors, positives, negatives = dist_weighted_sampling(labels,
                                                                      embeddings,
                                                                      high_var_threshold=high_var_threshold,
                                                                      nonzero_loss_threshold=nonzero_loss_threshold,
                                                                      neg_multiplier=neg_multiplier)
    if isinstance(beta_in, (float,int)):
        beta = beta_in
        beta_reg_loss = 0.0
    else:
        if isinstance(beta_in, tf.Tensor):
            assert tf.shape(beta_in).shape == 1
            k = tf.size(a_indices) / tf.size(labels)
            k = tf.cast(k, tf.int32)
            beta = tf.reshape(beta_in, (-1, 1))
            beta = tf.tile(beta, [1, k])
            beta = tf.reshape(beta, (-1,))
            beta_reg_loss = tf.reduce_sum(beta) * nu
        else:
            raise ValueError("`beta_in` must be one of [float, int, tf.Tensor].")

    d_ap = tf.sqrt(tf.reduce_sum(tf.square(positives - anchors), axis=1) + 1e-16)
    d_an = tf.sqrt(tf.reduce_sum(tf.square(negatives - anchors), axis=1) + 1e-16)

    pos_loss = tf.maximum(margin + d_ap - beta, 0)
    neg_loss = tf.maximum(margin + beta - d_an, 0)

    pair_cnt = tf.cast(tf.size(a_indices), tf.float32)

    # normalize based on the number of pairs
    loss = (tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss) + beta_reg_loss) / pair_cnt
    return loss


def distance_weighted_triplet_loss(labels, embeddings, margin=1.0, squared=False, high_var_threshold=0.5,
                                   nonzero_loss_threshold=1.4, neg_multiplier=1):
    """distance weighted sampling + triplet loss
    Args:
        labels: 1-D. tf.int32 `Tensor` with shape [batch_size] of multi-class integer labels.
        embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        margin: Float, margin term in the loss function.
        squared: Boolean, whether or not to square the triplet distances.
        nu: float. Regularization parameter for beta.
        high_var_threshold: float. cutoff for high gradient variance.
        nonzero_loss_threshold: float. cutoff for non-zero loss zone.
        neg_multiplier: int, default=1. the multiplier to enlarger the negative and positive samples.
    Returns:
        triplet_loss: tf.float32 scalar

    """
    a_indices, anchors, positives, negatives = dist_weighted_sampling(labels,
                                                                      embeddings,
                                                                      high_var_threshold=high_var_threshold,
                                                                      nonzero_loss_threshold=nonzero_loss_threshold,
                                                                      neg_multiplier=neg_multiplier)

    d_ap = tf.reduce_sum(tf.square(positives - anchors), axis=1)
    d_an = tf.reduce_sum(tf.square(negatives - anchors), axis=1)
    if not squared:
        d_ap = K.sqrt(d_ap + 1e-16)
        d_an = K.sqrt(d_an + 1e-16)

    loss = tf.maximum(d_ap - d_an + margin, 0)
    loss = tf.reduce_mean(loss)
    return loss
