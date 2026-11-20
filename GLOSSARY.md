# Glossary of suspicious letters

A human-readable reference for abbreviations encountered across computer
vision, machine learning, and the
[`cs231n-practice`](https://github.com/Maxstef/cs231n-practice) journey.

This is not intended to become a second textbook. Each entry should answer:
**what do the letters mean, what is the idea, and where does it normally
appear?** Some abbreviations have multiple meanings; those collisions are
marked explicitly because the letters were apparently running out.

## A–Z index

**A:** [AE](#ae--autoencoder), [AI](#ai--artificial-intelligence),
[AP](#ap--average-precision), [API](#api--application-programming-interface),
[AR](#ar--autoregressive), [ARO](#aro--attribution-relation-and-order)

**B:** [BC](#bc--behavioral-cloning), [BCE](#bce--binary-cross-entropy),
[BOS](#bos--beginning-of-sequence), [BPTT](#bptt--backpropagation-through-time)

**C:** [C3D](#c3d--3d-convolutional-network),
[CAM](#cam--class-activation-map), [CFG](#cfg--classifier-free-guidance),
[CIFAR-10](#cifar-10--canadian-institute-for-advanced-research-10),
[CLIP](#clip--contrastive-languageimage-pre-training),
[CLS](#cls--classification-token), [CNN](#cnn--convolutional-neural-network),
[COCO](#coco--common-objects-in-context), [CoCa](#coca--contrastive-captioner),
[CPC](#cpc--contrastive-predictive-coding), [CPU](#cpu--central-processing-unit),
[CREPE](#crepe--compositionality-evaluation-benchmark),
[CUDA](#cuda--compute-unified-device-architecture),
[CV](#cv--computer-vision)

**D:** [DAgger](#dagger--dataset-aggregation),
[DCGAN](#dcgan--deep-convolutional-generative-adversarial-network),
[DETR](#detr--detection-transformer), [DINO](#dino--self-distillation-with-no-labels),
[DiT](#dit--diffusion-transformer), [DP](#dp--data-parallelism),
[DQN](#dqn--deep-q-network)

**E:** [ELBO](#elbo--evidence-lower-bound),
[ELU](#elu--exponential-linear-unit), [EOS](#eos--end-of-sequence)

**F:** [FCN](#fcn--fully-convolutional-network),
[FFN](#ffn--feed-forward-network), [FID](#fid--fréchet-inception-distance),
[FLOP](#flop--floating-point-operation), [FN](#fn--false-negative),
[FP](#fp--false-positive), [FP32](#fp32--32-bit-floating-point),
[FPN](#fpn--feature-pyramid-network),
[FSDP](#fsdp--fully-sharded-data-parallel)

**G:** [GAN](#gan--generative-adversarial-network),
[GAP](#gap--global-average-pooling), [GELU](#gelu--gaussian-error-linear-unit),
[GPU](#gpu--graphics-processing-unit),
[Grad-CAM](#grad-cam--gradient-weighted-class-activation-mapping),
[GRU](#gru--gated-recurrent-unit)

**H:** [HSDP](#hsdp--hybrid-sharded-data-parallel)

**I:** [I3D](#i3d--inflated-3d-convnet),
[InfoNCE](#infonce--information-noise-contrastive-estimation),
[IoU](#iou--intersection-over-union), [IRL](#irl--inverse-reinforcement-learning)

**K:** [KL](#kl--kullbackleibler-divergence)

**L:** [L1](#l1--l1-norm-or-regularization),
[L2](#l2--l2-norm-or-regularization),
[LLaVA](#llava--large-language-and-vision-assistant), [LM](#lm--language-model),
[LSTM](#lstm--long-short-term-memory)

**M:** [MAE](#mae--masked-autoencoder-or-mean-absolute-error),
[mAP](#map--mean-average-precision), [MDP](#mdp--markov-decision-process),
[MFU](#mfu--model-flops-utilization), [MHA](#mha--multi-head-attention),
[ML](#ml--machine-learning), [MLE](#mle--maximum-likelihood-estimation),
[MLP](#mlp--multilayer-perceptron), [MoCo](#moco--momentum-contrast),
[MPC](#mpc--model-predictive-control), [MSE](#mse--mean-squared-error)

**N:** [NCHW](#nchw--batch-channel-height-width),
[NeRF](#nerf--neural-radiance-field), [NHWC](#nhwc--batch-height-width-channel),
[NLL](#nll--negative-log-likelihood), [NMS](#nms--non-maximum-suppression)

**P:** [PAD](#pad--padding-token), [PE](#pe--positional-encoding),
[PP](#pp--pipeline-parallelism), [PQ](#pq--panoptic-quality)

**Q:** [QKV](#qkv--query-key-value)

**R:** [R-CNN](#r-cnn--region-based-convolutional-neural-network),
[ReLU](#relu--rectified-linear-unit), [RGB](#rgb--red-green-blue),
[ResNet](#resnet--residual-network), [RL](#rl--reinforcement-learning),
[RMSE](#rmse--root-mean-squared-error),
[RNN](#rnn--recurrent-neural-network), [RoI](#roi--region-of-interest),
[RPN](#rpn--region-proposal-network), [RQ](#rq--recognition-quality)

**S:** [SDF](#sdf--signed-distance-function), [SGD](#sgd--stochastic-gradient-descent),
[SiLU](#silu--sigmoid-linear-unit),
[SimCLR](#simclr--simple-framework-for-contrastive-learning-of-visual-representations),
[SAM](#sam--segment-anything-model), [SQ](#sq--segmentation-quality),
[SSD](#ssd--single-shot-multibox-detector),
[SSL](#ssl--self-supervised-learning), [SVM](#svm--support-vector-machine)

**T:** [THW](#thw--time-height-width), [TP](#tp--true-positive-or-tensor-parallelism),
[TPU](#tpu--tensor-processing-unit)

**V:** [VAE](#vae--variational-autoencoder), [VGG](#vgg--visual-geometry-group),
[ViT](#vit--vision-transformer), [VLA](#vla--vision-language-action),
[VLM](#vlm--vision-language-model), [VQ](#vq--vector-quantization)

**X:** [XLA](#xla--accelerated-linear-algebra)

**Y:** [YOLO](#yolo--you-only-look-once)

## General ML, computing, and tensor notation

Representative sources: [Python and NumPy notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/00-python-numpy),
[distributed-training notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/11-distributed-training).

### AI — Artificial Intelligence

The broad field of building systems that perform tasks associated with
intelligent behavior. ML is one family of approaches inside AI.

### API — Application Programming Interface

A defined way for software components to communicate. A model API may provide
remote access to predictions without exposing weights, code, or training data.

### CIFAR-10 — Canadian Institute for Advanced Research 10

A small image-classification dataset with 10 classes and 32×32 RGB images. It
is popular for experiments because training finishes before everyone retires.

### CPU — Central Processing Unit

The general-purpose processor in a computer. Flexible and excellent at varied
work, but usually slower than GPUs for large parallel tensor operations.

### CUDA — Compute Unified Device Architecture

NVIDIA's platform and programming ecosystem for running parallel computation
on its GPUs. In PyTorch, moving tensors to a CUDA device normally means asking
an NVIDIA GPU to do the heavy lifting.

### CV — Computer Vision

The field concerned with extracting useful information and structure from
images, video, and other visual measurements.

### FP32 — 32-bit Floating Point

A common numerical format for model parameters and calculations. Lower-precision
formats can save memory and increase speed, but require more care with numerical
range and stability.

### GPU — Graphics Processing Unit

A highly parallel processor originally designed for graphics and now widely
used for matrix-heavy ML workloads.

### ML — Machine Learning

Methods that learn behavior or patterns from data instead of expressing every
decision as manually written rules.

### NCHW — Batch, Channel, Height, Width

A common image-tensor layout, especially in PyTorch. `N` is batch size and `C`
is the number of channels.

### NHWC — Batch, Height, Width, Channel

Another common image-tensor layout. Confusing it with NCHW produces models that
are less intelligent than the resulting error message suggests.

### RGB — Red, Green, Blue

A color representation using red, green, and blue channel intensities.

### THW — Time, Height, Width

The temporal and spatial dimensions of video features. A video tensor often
contains these alongside batch and channel dimensions.

## Losses, optimization, and classical classifiers

Representative sources: [linear-classifier notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/03-linear-classifiers),
[optimization notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/04-optimization).

### BCE — Binary Cross-Entropy

A loss for binary targets or independent yes/no labels. It penalizes predicted
probabilities that disagree with the target.

### L1 — L1 norm or regularization

The sum of absolute values. As a regularizer it often encourages sparse
parameters; as a distance it measures absolute coordinate differences.

### L2 — L2 norm or regularization

The Euclidean length, derived from squared values. L2 regularization discourages
large weights smoothly, while squared L2 distance heavily penalizes large errors.

### MLE — Maximum Likelihood Estimation

Choose parameters that make the observed training data as probable as possible
under the model.

### MSE — Mean Squared Error

The average squared difference between predictions and targets. Large errors
receive disproportionately large punishment.

### NLL — Negative Log-Likelihood

The negative logarithm of the probability assigned to the observed data.
Minimizing NLL is equivalent to maximizing likelihood and often produces more
numerically convenient sums.

### RMSE — Root Mean Squared Error

The square root of MSE. It returns the error to the original target's units,
which often makes the number easier to interpret.

### SGD — Stochastic Gradient Descent

Updates model parameters using gradients estimated from a minibatch rather
than the complete dataset. “Stochastic” means the update contains sampling
noise, not that the optimizer is acting on a whim.

### SVM — Support Vector Machine

A classifier that tries to separate classes with a large margin. The CS231n
multiclass version penalizes incorrect scores that approach or exceed the
correct-class score.

## Neural-network building blocks

Representative sources: [neural-network notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/05-neural-networks),
[convolutional-network notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/06-convolutional-networks).

### CAM — Class Activation Map

A coarse spatial map showing which final convolutional features support a
particular class. The original method requires a compatible architecture.

### CNN — Convolutional Neural Network

A network that applies learned local filters across spatial data. Shared
filters make CNNs efficient and naturally suited to images.

### ELU — Exponential Linear Unit

An activation that is linear for positive inputs and smoothly approaches a
negative saturation value for negative inputs.

### FCN — Fully Convolutional Network

A network that preserves spatial structure and produces spatial outputs rather
than collapsing everything into one classification vector. Common in semantic
segmentation.

### FFN — Feed-Forward Network

A small per-token neural network inside a Transformer block, commonly two
linear layers with an activation between them.

### GAP — Global Average Pooling

Averages each feature channel across all spatial positions, producing one
number per channel.

### GELU — Gaussian Error Linear Unit

A smooth activation widely used in Transformers. It softly gates values rather
than applying ReLU's hard zero boundary.

### Grad-CAM — Gradient-weighted Class Activation Mapping

Uses gradients of a class score to weight feature-map channels and produce a
coarse map of class-related evidence.

### MLP — Multilayer Perceptron

A feed-forward network made from learned affine layers and nonlinear
activations. Despite the name, modern usage does not require anyone to discuss
actual biological perceptrons.

### ReLU — Rectified Linear Unit

The activation `max(0, x)`. Simple, cheap, and capable of turning sufficiently
negative neurons into permanent decorative objects.

### ResNet — Residual Network

A network family built from residual blocks that learn a change to an identity
path. The shortcut gives gradients and information a less obstructed route
through deep models.

### SiLU — Sigmoid Linear Unit

The activation `x * sigmoid(x)`, also commonly called **Swish**. It is smooth
and permits small negative outputs.

### VGG — Visual Geometry Group

The Oxford research group whose name labels the VGG family of deep CNNs built
mainly from repeated small convolutions.

## Sequence models and Transformers

Representative sources: [sequence-model notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/07-sequence-models),
[attention and Transformer notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/08-attention-transformers).

### BOS — Beginning of Sequence

A special token marking where sequence generation begins.

### BPTT — Backpropagation Through Time

Training an RNN by unrolling it across sequence steps and propagating gradients
back through that unrolled computation.

### CLS — Classification token

A special Transformer token whose final representation is often used as the
summary input to a classifier. The brackets in `[CLS]` are traditional attire.

### EOS — End of Sequence

A special token indicating that sequence generation should stop.

### GRU — Gated Recurrent Unit

A gated RNN architecture that controls how much previous state to keep or
replace, using fewer gates than an LSTM.

### LM — Language Model

A model that assigns probabilities to text or predicts tokens, often one token
at a time.

### LSTM — Long Short-Term Memory

A gated RNN designed to preserve and update information over longer sequences
while reducing the gradient problems of a plain RNN.

### MHA — Multi-Head Attention

Several attention heads operating in parallel. Different heads can learn
different relationships before their outputs are combined.

### PAD — Padding token

A placeholder used to make variable-length sequences fit one rectangular
batch. A mask should prevent the model from treating padding as meaningful.

### PE — Positional Encoding

Information added to tokens so a Transformer can distinguish their positions.
Attention alone otherwise sees a set and has no natural sense of order.

### QKV — Query, Key, Value

The three representations used by attention. Queries ask what is relevant,
keys advertise what each position contains, and values carry the information
that is mixed.

### RNN — Recurrent Neural Network

A sequence model that repeatedly updates a hidden state while processing items
in order.

### ViT — Vision Transformer

A Transformer adapted to images, usually by splitting an image into patches
and treating those patches as a token sequence.

## Detection, localization, and segmentation

Representative sources: [detection notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/09-detection),
[segmentation notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/09-segmentation).

### AP — Average Precision

Summarizes a precision–recall curve for one class, usually at a chosen IoU
threshold. Exact interpolation rules depend on the evaluation protocol.

### COCO — Common Objects in Context

A large dataset and evaluation ecosystem for detection, segmentation,
keypoints, and captions. COCO AP averages results across several IoU thresholds.

### DETR — Detection Transformer

An object detector that predicts a fixed set of objects using a Transformer
and set-based matching, avoiding traditional anchor generation and NMS.

### FN — False Negative

A real target the system failed to detect: the object existed, but the model
politely denied all knowledge of it.

### FP — False Positive

A predicted target that does not correspond to a required real target.

### FPN — Feature Pyramid Network

Combines features at multiple spatial resolutions so a detector can represent
both small and large objects effectively.

### IoU — Intersection over Union

Overlap area divided by combined area for two masks or boxes. `1` means perfect
agreement; `0` means they have successfully avoided one another.

### mAP — mean Average Precision

An average of AP values, commonly across classes and sometimes across IoU
thresholds. Always check the protocol before comparing two mAP numbers.

### NMS — Non-Maximum Suppression

Keeps strong detections and removes weaker highly overlapping duplicates.

### PQ — Panoptic Quality

A panoptic-segmentation metric combining how accurately matched masks are
segmented (**SQ**) and how reliably instances are recognized (**RQ**).

### R-CNN — Region-based Convolutional Neural Network

A family of two-stage detectors that first considers candidate regions and
then classifies and refines them using CNN features.

### RoI — Region of Interest

A selected image or feature-map region requiring focused processing, such as a
candidate object box in a detector.

### RPN — Region Proposal Network

The component in Faster R-CNN that predicts candidate object regions from
shared convolutional features.

### RQ — Recognition Quality

The detection-like component of panoptic quality. It decreases with missed and
spurious instances.

### SAM — Segment Anything Model

A promptable segmentation model that produces masks from prompts such as
points, boxes, or existing masks. “Anything” is an ambition, not a warranty.

### SQ — Segmentation Quality

The average IoU of correctly matched segments in panoptic evaluation.

### SSD — Single Shot MultiBox Detector

A one-stage detector that predicts classes and boxes directly from dense
feature maps at several scales.

### TP — True Positive or Tensor Parallelism

In evaluation, a **true positive** is a prediction correctly matched to a real
target. In distributed training, **tensor parallelism** splits individual
tensor operations across devices. Context must perform the disambiguation.

### YOLO — You Only Look Once

A family of one-stage detectors designed to predict objects in a single dense
network pass.

## Self-supervised and representation learning

Representative sources: [self-supervised-learning notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/12-self-supervised).

### CPC — Contrastive Predictive Coding

Learns representations by predicting future or held-out latent information and
contrasting the correct target against alternatives.

### DINO — self-DIstillation with NO labels

A self-supervised approach in which a student learns from a slowly updated
teacher's predictions across different views of an image.

### InfoNCE — Information Noise-Contrastive Estimation

A contrastive objective that makes a positive pair similar relative to a set
of negative examples. Temperature controls how sharp the competition becomes.

### MAE — Masked Autoencoder or Mean Absolute Error

In self-supervised vision, a **Masked Autoencoder** reconstructs missing image
patches. In regression, **Mean Absolute Error** averages absolute prediction
errors. Same letters, unrelated paperwork.

### MoCo — Momentum Contrast

A contrastive-learning method using a slowly updated momentum encoder and a
queue of representations to provide many consistent negative examples.

### SimCLR — Simple Framework for Contrastive Learning of Visual Representations

Learns image representations by pulling together two augmented views of the
same image and pushing apart views from different images.

### SSL — Self-Supervised Learning

Learning where supervision is constructed from the data itself—for example by
hiding patches or pairing augmented views—rather than supplied as human labels.

## Generative models

Representative sources: [generative-model notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/13-generative-models),
[diffusion and GAN notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/14-generative-models).

### AE — Autoencoder

An encoder compresses an input into a latent representation and a decoder
reconstructs it. A plain AE is not automatically a probabilistic generative
model.

### AR — Autoregressive

A model that represents a joint distribution as a sequence of conditional
predictions, generating each element from earlier elements.

### CFG — Classifier-Free Guidance

A diffusion-generation technique that combines conditional and unconditional
predictions to strengthen adherence to a condition such as a text prompt.

### DCGAN — Deep Convolutional Generative Adversarial Network

A GAN architecture using convolutional generator and discriminator networks
for image generation. It established several practical architectural patterns
for making early image GANs train more reliably.

### DiT — Diffusion Transformer

A diffusion model whose denoising network uses a Transformer rather than the
traditional convolutional U-Net.

### ELBO — Evidence Lower Bound

The tractable objective used to train a VAE. It balances reconstruction with a
KL term that regularizes the approximate posterior toward the prior.

### FID — Fréchet Inception Distance

A generative-image metric comparing distributions of real and generated image
features. Lower is better, but the number depends strongly on data and protocol.

### GAN — Generative Adversarial Network

A generator tries to create realistic samples while a discriminator tries to
distinguish generated samples from real ones. Their disagreement becomes the
training signal and occasionally the source of considerable drama.

### KL — Kullback–Leibler divergence

A directional measure of how one probability distribution differs from
another. It is not a symmetric distance.

### VAE — Variational Autoencoder

A probabilistic latent-variable model trained with the ELBO. It learns both an
approximate posterior for encoding data and a decoder for generating samples.

### VQ — Vector Quantization

Maps a continuous representation to the nearest item in a finite learned
codebook, producing discrete latent tokens.

## Video and 3D vision

Representative sources: [video-understanding notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/10-video-understanding),
[3D-vision notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/15-3d-vision).

### C3D — 3D Convolutional Network

A video architecture that applies 3D convolutions across time, height, and
width to learn motion and appearance jointly.

### I3D — Inflated 3D ConvNet

Builds a 3D video network by expanding successful 2D convolutional filters
along the time dimension, allowing reuse of image-pretrained weights.

### NeRF — Neural Radiance Field

A neural scene representation mapping 3D positions and viewing directions to
density and color, enabling novel-view rendering through volume integration.

### SDF — Signed Distance Function

A function giving distance to a surface, with the sign indicating which side
of the surface a point occupies. The zero level set defines the surface.

## Vision and language

Representative sources: [vision-language notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/16-vision-language).

### ARO — Attribution, Relation, and Order

A benchmark suite testing whether a vision-language model understands
attribute bindings, relations, and word order rather than merely noticing a
bag of familiar concepts.

### CLIP — Contrastive Language–Image Pre-training

Learns aligned image and text embeddings from paired data. It supports
retrieval and zero-shot classification by comparing images with text prompts.

### CoCa — Contrastive Captioner

A vision-language model combining contrastive image–text alignment with an
autoregressive captioning objective, so retrieval and generation share one
system.

### CREPE — compositionality evaluation benchmark

A vision-language benchmark testing whether models can recombine familiar
concepts systematically and handle descriptions of increasing compositional
complexity. Here CREPE is a benchmark name; the paper title asks whether
vision-language foundation models can reason compositionally.

### LLaVA — Large Language and Vision Assistant

A family of multimodal assistants that connects visual features to a language
model for instruction-following conversations about images.

### VLM — Vision-Language Model

A model that combines visual and linguistic information for tasks such as
retrieval, captioning, visual question answering, or multimodal conversation.

## Robot learning and control

Representative source: [robot-learning notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/17-robot-learning).

### BC — Behavioral Cloning

Supervised imitation learning that predicts an expert's action from the
observed state. It can fail after small mistakes lead to unfamiliar states.

### DAgger — Dataset Aggregation

An imitation-learning method that runs the learner, asks an expert to label the
states the learner actually visits, and repeatedly adds those corrections to
the training data.

### DQN — Deep Q-Network

A neural network that estimates action values for Q-learning, commonly trained
with replay data and a separate target network.

### IRL — Inverse Reinforcement Learning

Infers a reward function that could explain expert demonstrations, then uses
that recovered objective to learn behavior.

### MDP — Markov Decision Process

A mathematical description of sequential decision-making using states,
actions, transitions, rewards, and a discount factor.

### MPC — Model Predictive Control

Repeatedly plans a short action sequence using a dynamics model, executes the
first action, observes the result, and plans again.

### RL — Reinforcement Learning

Learning behavior through interaction and reward, including the inconvenience
that actions change which data will be observed next.

### VLA — Vision-Language-Action

A model that conditions robot actions on visual observations and language
instructions.

## Distributed training and hardware

Representative source: [distributed-training notes](https://github.com/Maxstef/cs231n-practice/tree/main/notes/11-distributed-training).

### DP — Data Parallelism

Each worker holds a model replica and processes different data; gradients are
then synchronized. Do not confuse this local glossary meaning with dynamic
programming, which has stolen the same letters elsewhere.

### FLOP — Floating-Point Operation

One arithmetic operation on floating-point values. **FLOPs** may mean either a
count of operations or operations per second, so surrounding units matter.

### FSDP — Fully Sharded Data Parallel

Data-parallel training that shards parameters, gradients, and optimizer state
across workers to reduce the persistent memory burden on each device.

### HSDP — Hybrid Sharded Data Parallel

Uses sharding inside well-connected device groups and replication across
groups, matching communication strategy to cluster topology.

### MFU — Model FLOPs Utilization

The fraction of theoretical peak compute that useful model operations achieve.
Communication, data loading, and inefficient kernels lower it.

### PP — Pipeline Parallelism

Splits consecutive groups of model layers across devices and moves
microbatches through them like stages of a pipeline.

### TPU — Tensor Processing Unit

Google's accelerator designed for large tensor operations used in ML.

### XLA — Accelerated Linear Algebra

A compiler stack that optimizes tensor computation graphs for accelerators,
notably TPUs and supported GPU/CPU backends.

## Maintenance rule

Add a term when it causes confusion or becomes important in a practice. Keep
the explanation short, place it in the relevant topic, and add it to the A–Z
index. If one entry starts demanding diagrams, equations, and its own emotional
support system, it probably deserves a notebook instead.
