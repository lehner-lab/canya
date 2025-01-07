import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Activation,
    ActivityRegularization,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Layer,
    LayerNormalization)
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
from pkg_resources import resource_filename


seqminsize=24
paddings=tf.constant([[2,2],[0,0]])

def dynamic_padding(inp, min_size,
                   consval=-1,post=True):
    pad_size = min_size - tf.shape(inp)[0]
    paddings = [[0, pad_size], [0, 0]] # assign here, during graph execution
    if not post:
        paddings = [[pad_size, 0], [0, 0]]
    return tf.pad(inp, paddings,constant_values=consval)

aas="ACDEFGHIKLMNPQRSTVWY"
def str_to_vector(str, template):
    #   return [ei_vec(template.index(nt),len(template)) for nt in str]
    mapping = dict(zip(template, range(len(template))))
    seq = [mapping[i] for i in str]
    return np.eye(len(template))[seq]

def seq_to_vector(seq):
    return str_to_vector(seq, aas)

# Loss function from Regev Lab interpretable splicing model:
# https://github.com/regev-lab/interpretable-splicing-model
def binary_KL(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_pred, y_true)-K.binary_crossentropy(y_true, y_true), axis=-1)   # this is for the Ubuntu machine in Courant
    return tf.keras.backend.mean(
        tf.keras.backend.binary_crossentropy(y_true, y_pred)
        - tf.keras.backend.binary_crossentropy(y_true, y_true),
        axis=-1,
    )

class attnmaskbin(tf.keras.layers.Layer):
    def call(self, x):
        return tf.cast(x,tf.float32,name="masked_input_binary")

class getattnmask(tf.keras.layers.Layer):
  def __init__(self, num_repeats,name="attn_mask"):
    super(getattnmask, self).__init__()
    self.num_repeats = num_repeats

  def call(self, inputs):
    sequence_masks_attn=tf.squeeze(inputs)
    sequence_masks_attn=tf.abs(tf.subtract(inputs,tf.constant(1,dtype=tf.float32)))
    sequence_masks_attn=tf.repeat(sequence_masks_attn,repeats=sequence_masks_attn.shape[-2],
             axis=-1)
    sequence_masks_attn=tf.transpose(sequence_masks_attn,[0,2,1])
    sequence_masks_attn=sequence_masks_attn[:,tf.newaxis,:,:]
    return sequence_masks_attn


## Class (and member functions) written by Chandana Rajesh and Peter Koo at CSHL
class MultiHeadAttention_wpos(keras.layers.Layer):
    def __init__(self, d_model, num_heads, embedding_size=None,name="MHA"):
        super(MultiHeadAttention_wpos, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size == None else embedding_size

        assert d_model % self.num_heads == 0 and d_model % 6 == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model, use_bias=False)
        self.wk = keras.layers.Dense(d_model, use_bias=False)
        self.wv = keras.layers.Dense(d_model, use_bias=False)

        self.r_k_layer = keras.layers.Dense(d_model, use_bias=False)
        self.r_w = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True,
                              name="r_w")
        self.r_r = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True,
                              name="r_r")

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)
        q = q / tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32))

        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size=self.embedding_size//6

        seq_length = tf.cast(seq_len, dtype=tf.float32)
        exp1 = f_exponential(tf.abs(pos), feature_size, seq_length=seq_length)
        exp2 = tf.multiply(exp1, tf.sign(pos)[..., tf.newaxis])
        cm1 = f_central_mask(tf.abs(pos), feature_size, seq_length=seq_length)
        cm2 = tf.multiply(cm1, tf.sign(pos)[..., tf.newaxis])
        gam1 = f_gamma(tf.abs(pos), feature_size, seq_length=seq_length)
        gam2 = tf.multiply(gam1, tf.sign(pos)[..., tf.newaxis])

        # [1, 2seq_len - 1, embedding_size]
        positional_encodings = tf.concat([exp1, exp2, cm1, cm2, gam1, gam2], axis=-1)
        positional_encodings = keras.layers.Dropout(0.1)(positional_encodings)

        # [1, 2seq_len - 1, d_model]
        r_k = self.r_k_layer(positional_encodings)

        # [1, 2seq_len - 1, num_heads, depth]
        r_k = tf.reshape(r_k, [r_k.shape[0], r_k.shape[1], self.num_heads, self.depth])
        r_k = tf.transpose(r_k, perm=[0, 2, 1, 3])
        # [1, num_heads, 2seq_len - 1, depth]

        # [batch_size, num_heads, seq_len, seq_len]
        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)

        # [batch_size, num_heads, seq_len, 2seq_len - 1]
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        # [batch_size, num_heads, seq_len, seq_len]
        relative_logits = relative_shift(relative_logits)

        # [batch_size, num_heads, seq_len, seq_len]
        logits = content_logits + relative_logits
        #### add in masking capability ####
        logits = tf.where(tf.equal(mask, 1), -10000.0, logits)
        attention_map = tf.nn.softmax(logits)

        # [batch_size, num_heads, seq_len, depth]
        attended_values = tf.matmul(attention_map, v)
        # [batch_size, seq_len, num_heads, depth]
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attended_values, [batch_size, seq_len, self.d_model])

        output = self.dense(concat_attention)

        return output, attention_map



#------------------------------------------------------------------------------------------
# Positional encoding functions for Multi-Head Attention
#------------------------------------------------------------------------------------------
## Class (and member functions) written by Chandana Rajesh and Peter Koo at CSHL
def f_exponential(positions, feature_size, seq_length=None, min_half_life=3.0):
    if seq_length is None:
        seq_length = tf.cast(tf.reduce_max(tf.abs(positions)) + 1, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = tf.reshape(half_life, shape=[1]*positions.shape.rank + half_life.shape)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    return outputs

def f_central_mask(positions, feature_size, seq_length=None):
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32)) - 1
    center_widths = tf.reshape(center_widths, shape=[1]*positions.shape.rank + center_widths.shape)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    return outputs

def f_gamma(positions, feature_size, seq_length=None):
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    stdv = seq_length / (2*feature_size)
    start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = tf.reshape(mean, shape=[1]*positions.shape.rank + mean.shape)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2
    def gamma_pdf(x, conc, rt):
        log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
        log_normalization = (tf.math.lgamma(concentration) - concentration * tf.math.log(rate))
        return tf.exp(log_unnormalized_prob - log_normalization)
    probabilities = gamma_pdf(tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis], concentration, rate)
    outputs = probabilities / tf.reduce_max(probabilities)
    return outputs

def relative_shift(x):
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x







def create_model(
    indim=20,
    input_length=149,    
    dense_number=64,
    num_filters=100,
    filter_width=3,
    dropout_rate=0.4,
    num_heads=1,
    key_len=6,
    l2_regularization=0.01,
    pool_size="no",
    batch_size=256,
    act_reg=0
):
########################
    ## Define model logic ##
    ########################
    
    # Inputs
    seq_input = Input(shape=(input_length, indim),name="seq_input")
    
    # Get masked input, mask objects
    seq_input_masked=keras.layers.Masking(mask_value=-1,name="masked_input")(seq_input)
    ## Actual binary masks
    sequence_masks=keras.layers.Masking(mask_value=-1,name="computed_masked_input").compute_mask(seq_input)
    sequence_masks_bin=tf.cast(sequence_masks,tf.float32,name="masked_input_binary")
    # sequence_masks_bin=attnmaskbin()(sequence_masks)
    ## Attention masks
    if pool_size=="no":
        sequence_masks_attn=getattnmask(input_length)(sequence_masks_bin[:,:,tf.newaxis])
    elif isinstance(pool_size,int):
        sequence_masks_attn=keras.layers.MaxPool1D(pool_size=pool_size,name="attn_mask_util")(sequence_masks_bin[:,:,tf.newaxis])
        sequence_masks_attn=getattnmask(sequence_masks_attn.shape[1],name="attn_mask_out")(sequence_masks_attn)
    
    # Sequence conv
    primary_conv=Conv1D(filters=num_filters, kernel_size=filter_width, name="seq_conv",
                       padding="same",use_bias=False)(seq_input_masked)
    out_seq_conv=Activation("exponential",name="seq_conv_activation")(primary_conv)
    
    # induce sparisty before or after?
    out_seq_conv=keras.layers.ActivityRegularization(l1=act_reg)(out_seq_conv)
    
    # After activation, mask to 0
    out_seq_conv=tf.keras.layers.Multiply(name="seq_conv_mask")([out_seq_conv,sequence_masks_bin[:,:,tf.newaxis]])
    
    # Pooling?
    if isinstance(pool_size,int):
        out_seq_conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(out_seq_conv)
    
    # Dropout of convolution
    out_seq_conv=Dropout(.1,name="seq_conv_act_dropoout")(out_seq_conv)
    
    # Pass in the attention with the masks
    out_seq_attn, wseq=MultiHeadAttention_wpos(num_heads=num_heads,
                                                d_model=num_heads*key_len,name="MHA")(
        out_seq_conv,out_seq_conv,out_seq_conv,
        sequence_masks_attn)
    
    # Dropout of attention
    out_seq_attn=Dropout(.1,name="seq_attn_dropout")(out_seq_attn)
    out_seq_attn=Flatten(name="Flatten")(out_seq_attn)
    
    

    ### low-rank weight matrices can lead to redundant filters
    ### https://people.cs.umass.edu/~arunirc/downloads/pubs/redundant_filter_dltp2017.pdf
    ### if this is true, test with and without?
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(out_seq_attn)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    out=Dense(1,activation="sigmoid",name="output_activation")(dense)
    # create model
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer="adam", loss=binary_KL)
    # model.compile(optimizer="adam", loss="bce")
    
    return model

def get_canya(modweights="models/model_weights.h5"):
    canyamodel=create_model()
    canyamodel.load_weights(resource_filename(__name__,modweights))
    print("Loaded model " + modweights)
    return canyamodel

def get_predictions(model,sequences):
    embeddedseqs=[tf.pad(seq_to_vector(x),paddings,"CONSTANT") for x in sequences]
    embeddedseqs=[dynamic_padding(x,seqminsize,post=True) for x in embeddedseqs]
    embeddedseqs=np.asarray(embeddedseqs)
    curpredictions=model.predict(embeddedseqs)
    return curpredictions.flatten().tolist()
    








    