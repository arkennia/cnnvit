import tensorflow as tf
from tensorflow import keras
from typing import List
from keras import layers


class AttentionBlock(layers.Layer):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            attention_mask: tf.Tensor):
        super.__init__(self, AttentionBlock)
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_mask = attention_mask

        self.attn = layers.MultiHeadAttention(num_heads, d_model, dropout=0.1)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-5)
        self.mlp = keras.Sequential([layers.Dense(
            d_model, activation=keras.activations.gelu()), layers.Dropout(0.1)])
        self.ln_2 = layers.LayerNormalization(epislon=1e-5)

    def call(self, x: tf.Tensor):
        ln_x = self.ln_1(x)
        x = x + self.attn(ln_x, ln_x, attention_mask=self.attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(layers.Layer):
    def __init__(
            self,
            width: int,
            layers: int,
            num_heads: int,
            attn_mask: tf.Tensor = None):
        super.__init__(self, Transformer)
        self.width = width
        self.layers = layers
        self.heads = [
            AttentionBlock(
                width,
                num_heads,
                attn_mask) for _ in range(layers)]

    def call(self, x: tf.Tensor):
        for i in range(layers):
            x = self.heads[i](x)
        return x


class VisionTransformer(keras.Model):
    def __init__(
            self,
            input_resolution: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            output_dim: int):
        super.__init__(self, VisionTransformer)
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # Patch creation
        self.conv_1 = layers.Conv2D(
            width, patch_size, patch_size, use_bias=False)
        scale = width ** -0.5
        self.class_embedding = tf.Variable(
            scale * tf.random.normal(width), trainable=True)
        self.positional_embedding = tf.Variable(
            scale *
            tf.random.normal(
                (input_resolution //
                 patch_size) ** 2 +
                1,
                width),
            trainable=True)
        self.ln_pre = layers.LayerNormalization(epsilon=1e-5)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = layers.LayerNormalization(epsilon=1e-5)
        self.proj = tf.Variable(scale * tf.random.uniform([width, output_dim]))

    def call(self, x: tf.Tensor):
        x = self.conv_1(x)
        x = tf.reshape(x, [x.shape[0], -1, x.shape[-1]])
        x = tf.concat(
            [self.class_embedding + tf.zeros(x.shape[0], 1, x.shape[-1]), x], axis=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj
        return x


class ConvNet(keras.Model):
    def __init__(self, filters, kernel_sizes, dropout):
        super.__init__(self, ConvNet)
        self.model = keras.Sequential()
        for size in kernel_sizes:
            self.model.add(layers.Conv1D(
                filters,
                size,
                activation="relu",
                kernel_constraint=keras.constraints.MaxNorm(3)
            )
            )
            self.model.add(layers.Dropout(dropout))
        self.model.add(layers.GlobalMaxPool1D())

    def call(self, x: tf.Tensor):
        return self.model(x)


class Model(keras.Model):
    def __init__(self, embed_dim: int, filters: int, kernel_sizes: List[int],
                 cnn_dropout: float, image_resolution: int,
                 vision_layers: int, vision_width: int,
                 vision_patch_size: int):
        super.__init__(self, Model)
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )
