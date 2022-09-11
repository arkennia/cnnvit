# CNN Diffusion
A Tensorflow model inspired by Stable Diffusion. 

## Description

This model is inspired by Stable Diffusion published by CompViz.  

It uses the following architecure:  
* The text encoder is based on CLIP by OpenAI, however rather than a transformer, it uses a CNN for the text encoding.
* The image encoder is a Visual Transformer, ported from the OpenAI CLIP PyTorch code.
