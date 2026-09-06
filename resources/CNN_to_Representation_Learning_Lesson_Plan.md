# CNN → Representation Learning --- Lesson Plan

## Lesson 1 --- From Classification to Structured Prediction

**Central question:** *A CNN has learned useful features. Why should it
answer only one question?*

### 1. Classification is just Backbone + Head

**Slides** - Recall transfer learning / CNN as feature extractor -
Introduce terminology: **backbone → representation → head** - Original
classifier is just one possible head

**Infographic** - Encoder/backbone as a general reusable feature
extractor

Keep this short --- it is mostly reframing something students already
know.

### 2. One Backbone, Multiple Heads

**Slides** - Shared representation → multiple outputs - Example: object
class + animal/vehicle - Different heads can solve different types of
problems - Multiple losses:

$$L=L_1+\alpha L_2$$

-   Both losses backpropagate through the shared backbone

**Colab: `One_Backbone_Two_Heads_Exercise.ipynb`**

Place this exactly here.

Progression: - pretrained ResNet-18 - remove classifier - shared feature
vector - two heads - two targets - two losses - vary $\alpha$ - inspect
gradient paths

Keep the $\alpha=0,1,10$ experiment. It turns multi-head learning from
an architectural diagram into an understanding of how multiple
objectives compete/cooperate through a shared representation.

**Transition question:**\
*What if one head says WHAT, while another says WHERE?*

### 3. Classification → Localization → Detection

**Slides** - Classification: What? - Localization: What + Where? -
Bounding-box regression - Multiple objects make the problem harder - Why
spatial information matters - Callback to GAP: classification can
discard location; detection cannot

**Infographics** - Classification → Detection - Why Detection Cannot
Throw Away "Where"

### 4. Naive Detection: Sliding Window

**Slides** - "We already own a classifier. Can we build a detector?" -
Sliding window - Different window positions/scales - Huge repeated
computation

This is important historically because students should be able to invent
it themselves.

**Possible small Colab/demo** - Take a pretrained classifier - Crop
several regions from one image - Classify each crop - Show the repeated
computation

A short visual demo is enough; it does not need to become an exercise.

### 5. R-CNN: Stop Looking Everywhere

**Slides** - Region proposals - Crop proposed regions - CNN feature
extraction - Classification - Bounding-box regression - Explicit
connection back to transfer learning

Then introduce conceptually: - R-CNN → Fast R-CNN → Faster R-CNN - Do
not teach every historical architecture in detail

**Colab: Fast R-CNN / RoI Pooling**

Place the RoI Pooling notebook here, after R-CNN and while introducing
Fast R-CNN.

Teaching question:

> R-CNN runs the CNN repeatedly. Could we run the CNN **once on the
> whole image** and then extract features for each region?

Use RoI Pooling to demonstrate **shared computation**, without turning
it into a long technical detour.

### 6. Bounding Boxes and IoU

**Slides** - Ground-truth vs predicted box - Intersection - Union -
IoU - Good/poor predictions - Clarify overlap/evaluation measure vs
training loss

**Suggested Colab** Interactive IoU demo with two movable/slidable boxes
and continuously updated IoU.

Students can explore: - same center but wrong size - correct size but
displaced - partial overlap - no overlap

### 7. Modern Detection → YOLO

**Slides** - The one-stage idea - Image once → feature maps → many
predictions - Detection heads - class + box + confidence/objectness -
multi-scale prediction - NMS - YOLO evolution as a short architecture
timeline, not version trivia

**Colab/demo** Run a pretrained modern YOLO detector on: 1. ordinary
image 2. crowded image 3. unusual/difficult image 4. webcam/video if
practical

Expose confidence threshold and NMS/IoU threshold interactively so
students can see duplicate boxes appear/disappear.

### 8. Deployment Payoff --- YOLO in the Browser

**Slides/demo** - inference vs training - model export - browser/edge
inference - latency - model size - callback to Network Optimization
techniques

This gives Lesson 1 a real-world engineering ending.

------------------------------------------------------------------------

# Lesson 2 --- Encoder--Decoder & Learning Without Labels

**Central question:** *Can a network learn useful representations
without humans labeling the data?*

## 1. Detection → Segmentation

**Slides** - Classification = one label - Detection = objects +
approximate locations - Segmentation = label every pixel - Medical
segmentation as motivating example

**Infographic** - One Label → One Box → Every Pixel

## 2. U-Net: Encoder → Decoder

**Slides** - Why segmentation needs spatial output - Contracting path =
encoder - Bottleneck - Expanding path = decoder - Skip connections - Why
skip connections restore fine spatial information

**Suggested Colab** A small U-Net segmentation demonstration emphasizing
tensor shapes rather than expensive training.

Example progression:

``` text
256×256
  ↓
128×128
  ↓
64×64
  ↓
32×32
  ↓
64×64
  ↓
128×128
  ↓
256×256
```

Display which encoder feature map is concatenated at each skip
connection.

## 3. The Critical Transition: What Else Can a Decoder Produce?

**Slides**

Start with:

``` text
Image → Encoder → Representation → Decoder → Segmentation
```

Then change exactly one thing:

``` text
Image → Encoder → Representation → Decoder → Image
```

Ask:

> What if the target is simply the original input?

**Autoencoder.**

## 4. Why Would We Do That?

Make the labeling-cost argument central.

**Slides** - Supervised learning requires human-provided targets -
Annotation is expensive - Raw images/text/audio are abundant - Can data
provide its own training signal?

``` text
Supervised:
image → human → CAT

Autoencoder:
image → network → image
                  ↑
             target already exists
```

**Infographics** - Labeling Cost Problem - One Dataset, Different
Targets

Introduce **self-supervised learning** prominently.

## 5. Classic Autoencoder

**Slides** - Encoder - latent representation - bottleneck - decoder -
reconstruction loss - connection back to PCA/dimensionality reduction -
nonlinear learned representation

**Suggested Colab** MNIST autoencoder with a deliberately tiny 2D latent
space.

Use it to explore: - reconstruction - bottleneck - latent
visualization - clustering - interpolation

This connects directly to the earlier dimensionality-reduction lesson.

## 6. "Break It and Fix It"

**Slides** - Why merely copying input can be too easy - Corrupt input,
preserve clean target - denoising - inpainting - super-resolution -
masking

``` text
clean x
   ↓ automatically corrupt
corrupt(x) → network → x
```

**Suggested Colab** MNIST or CIFAR with random rectangles erased. Train
reconstruction from corrupted → clean.

## 7. Masked Autoencoders → Self-Supervised Pretraining

**Slides** - Mask patches automatically - Encoder sees incomplete
information - Decoder reconstructs - Reconstruction is not necessarily
the final goal - **The encoder is the prize** - discard/ignore decoder -
fine-tune encoder downstream

## 8. Same Trick in NLP

**Slides**

``` text
"The cat sat on the mat"
          ↓
"The cat [MASK] on the mat"
          ↓
        predict SAT
```

Nobody manually labeled `SAT`; the training problem was manufactured
from existing data.

Side-by-side analogy:

``` text
IMAGE                       LANGUAGE

mask patches                mask words
     ↓                           ↓
predict missing content     predict missing token
     ↓                           ↓
learn representation        learn representation
```

Mention BERT-style masked language modeling here.

The important general concept is **self-supervised learning**, rather
than an autoencoder-specific trick.

## 9. Autoencoder → VAE

**Slides** - Problem with arbitrary AE latent spaces - Want a
structured/smooth latent space - distributions rather than isolated
encodings - sampling - interpolation - generation

Use the existing MNIST latent-space explorer.

**Suggested Colab** Interactive 2D MNIST VAE latent-space explorer:

``` text
(z₁, z₂) → Decoder → digit
```

Allow students to move through latent space and observe decoded digits.

## 10. Conditional VAE

**Slides** - Latent representation + condition - MNIST digit condition -
one-hot encoding - controlled generation

Then ask:

> Suppose the dataset contained **face + age**. Could age become the
> condition?

``` text
         z ───────────┐
                      ↓
age = 20 ───────→ Decoder → young face

         same z ──────┐
                      ↓
age = 70 ───────→ Decoder → older face
```

This becomes the first introduction to **controllable generative
models** and foreshadows richer conditioning later.

------------------------------------------------------------------------

# Lesson 3 --- Feature-Space Objectives

This material should begin the next arc rather than being squeezed into
Lesson 2.

**Central question:** *Once we have learned representations, can we use
them to define what "similar" means?*

## 1. Pixel Similarity ≠ Perceptual Similarity

**Colab: `Perceptual_Loss_STL10_Demo2.ipynb`**

Use this as the opening experiment.

It compares: - pixel MSE reconstruction - MSE + VGG perceptual loss -
interactive $\lambda$

**Transition question:**

> We've been judging reconstruction by comparing pixels. But is pixel
> similarity actually the same as perceptual similarity?

## 2. Learned Features as Similarity

**Slides** - compare feature activations instead of only pixels -
pretrained CNN as perceptual feature extractor - content representation

## 3. What About Style?

**Colab: `Gram_Matrices_StyleTransfer_Part1.ipynb`**

Place this after perceptual loss.

Progression: - fixed pretrained VGG-19 - extract feature maps - Gram
matrices - compare style representations

Then continue into Neural Style Transfer.

``` text
Autoencoder / VAE
        ↓
"What does similar mean?"
        ↓
Perceptual_Loss_STL10_Demo2
        ↓
Learned features as similarity
        ↓
Content representation
        ↓
"What about style?"
        ↓
Gram_Matrices_StyleTransfer_Part1
        ↓
Neural Style Transfer
```

------------------------------------------------------------------------

# Compact Teaching Map

  -------------------------------------------------------------------------------------------
  Lesson                  Flow                    Existing notebooks
  ----------------------- ----------------------- -------------------------------------------
  **1. Structured         Backbone/head →         `One_Backbone_Two_Heads_Exercise.ipynb`
  Prediction**            multi-head →            early; Fast R-CNN / RoI Pooling during Fast
                          localization → sliding  R-CNN
                          window → R-CNN →        
                          Fast/Faster R-CNN → IoU 
                          → YOLO → browser        
                          deployment              

  **2. Learning Without   Segmentation → U-Net →  New small AE/MAE/VAE demos recommended
  Labels**                encoder/decoder →       
                          Autoencoder → labeling  
                          problem →               
                          self-supervision →      
                          Break It & Fix It → MAE 
                          → NLP masking → VAE →   
                          Conditional VAE         

  **3. Feature-Space      Pixel-loss limitations  `Perceptual_Loss_STL10_Demo2.ipynb` →
  Objectives**            → perceptual loss →     `Gram_Matrices_StyleTransfer_Part1.ipynb`
                          feature similarity →    
                          Gram matrices → style → 
                          Neural Style Transfer   
  -------------------------------------------------------------------------------------------

## Questions Students Should Remember

**Lesson 1:**\
*What else can I attach to a learned representation?*

**Lesson 2:**\
*How can I learn the representation without paying humans to label
everything?*

**Lesson 3:**\
*Once I have learned representations, can I use them to define what
"similar" means?*
