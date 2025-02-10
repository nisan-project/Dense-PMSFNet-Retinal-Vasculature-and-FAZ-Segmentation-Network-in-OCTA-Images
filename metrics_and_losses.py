import tensorflow as tf
from tensorflow.keras import backend as K

'''
metrics and losses
'''

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    epsilon = 1.e-5
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    logits = tf.math.log(y_pred / (1 - y_pred))
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=logits)
    bce = tf.reduce_mean(bce)
    return bce

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25,epsilon=1e-5):
    pt = y_true * y_pred
    gt = 1.0 - y_true
    p_t = tf.where(K.equal(y_true, 1), pt, gt)
    alpha_t = tf.where(K.equal(y_true, 1), alpha, 1.0 - alpha)
    ce_loss = -K.log(p_t + epsilon)
    weight = K.pow((1 - p_t), gamma)
    focal_loss = weight * alpha_t * ce_loss
    return focal_loss

def total_loss(y_true, y_pred):
    dice_loss_value = dice_loss(y_true, y_pred)
    focal_loss_value = focal_loss(y_true, y_pred)
    total_loss_value = dice_loss_value + focal_loss_value
    return total_loss_value

def accuracy(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(K.round(y_pred))
    correct = K.sum(K.cast(K.equal(y_true, y_pred), 'float32'))
    total = K.sum(K.cast(K.equal(y_true, y_true), 'float32'))  
    epsilon = K.epsilon()
    accuracy = correct / (total + epsilon)
    return accuracy

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(
    K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = intersection / union
    return iou


