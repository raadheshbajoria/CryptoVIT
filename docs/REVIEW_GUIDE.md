# Review guide (Phase 1 capstone review)

Numbers quoted here were measured on the held-out DermaMNIST test set (2,005 images), one training
run per row (seed 42). The full table, including every row, is in `RESULTS.md`. Research prototype;
not a medical device; not for clinical use.

## One-sentence pitch

A skin-lesion image is scrambled with a secret key and a Vision Transformer classifies the scrambled
image **without ever decrypting it**; we measure how much accuracy that costs and how the choice of
scrambling decides it.

## Slides (about 8-10 minutes)

1. **Problem.** Medical images are sensitive. Normally a cloud AI sees the raw image. Idea: scramble
   it on the sender's side, classify the scrambled image, never decrypt on the server.
2. **Pipeline.** image -> encrypt with key -> enlarge -> ViT (196 patches of 16x16) -> one of 7 classes.
   Show `presentation/safe_melanoma_test846_side_by_side.png`.
3. **Dataset.** DermaMNIST: 7 lesion classes, 28x28 images, 7,007 / 1,003 / 2,005 split. 67% of the
   training images are benign nevi, so we report balanced accuracy and per-class sensitivity.
4. **Encryption and the key.** The key is a secret string. It is hashed (SHA-256) into the starting
   state of a 4-dimensional chaotic system; a tiny change in the key gives a completely different
   scrambling. Two kinds of scheme: per-pixel (strong scrambling, 4D chaos + DNA-XOR) and block-wise
   (shuffles 2x2 blocks plus a few reversible operations). Both decrypt exactly.
5. **Models.** A plain ViT on plaintext (baseline); the same ViT on encrypted images; the ViT with our
   `CryptoPatchEmbedding`; and `CryptoViT` with the key-aware positional encoding.
6. **Result 1: per-pixel scrambling destroys what the model can learn.** Plaintext model 53.3% balanced
   accuracy; the same model on per-pixel-encrypted images 19.4% (chance is 14.3%); with
   `CryptoPatchEmbedding` 23.9%.
7. **Result 2: block-wise scrambling keeps it learnable.** Block-encrypted plain ViT 53.6% (AUC 0.809),
   with `CryptoPatchEmbedding` 53.0% (AUC 0.839), `CryptoViT` 54.9% (AUC 0.815, melanoma sensitivity
   71.3% vs 64.1% for the plaintext model). **All three are within a few points of each other and of the
   plaintext model (53.3%), so the big gain over per-pixel scrambling comes from the block scheme, not
   from the custom components - say this yourself.** Why block works: a 2x2 block is exactly one
   16x16 patch after enlargement, and the scrambling steps are things a ViT can absorb (patch embedding
   and learnable positions).
8. **Honest security picture.** Show the side-by-side strip: per-pixel is pure noise; block keeps the
   colour palette up to inversion and has no diffusion. Quote `SECURITY.md`: changing one pixel
   changes 20.5% of the output for per-pixel and 0.04% (one pixel) for block, against about 99.6% for
   a good cipher. We trade security for classifiability.
9. **Live demo** (below).
10. **Limitations and Phase 2.** See the end of this file.

## Live demo (about 3 minutes)

Start it with `python scripts/start_demo.py --block 2` (uses the block-scheme models).

1. Upload `demo_images/1_safe_melanoma_test846.png` in the dashboard. The original appears enlarged.
2. Point at the encrypted image: the lesion is unrecognisable. Mention the key hash under it.
3. Read the table. For this image all four models say melanoma: plaintext A 62%, and the three that
   only saw the encrypted image B 57%, C 50%, D 65% ("All trained models agree" appears under the
   table). Each model is wrong on some other images, so do not call it perfect.
4. Click **Decrypt with the correct key**: "exactly identical to the original".
5. Click **Decrypt with a wrong key**: the image is not recovered. With the block scheme it comes out
   as a different scrambled mosaic of the same colours (not pure noise, but unrecognisable).
6. Optional: tick "Also classify with the wrong key" in the sidebar to show that the encrypted-image
   models then become unreliable.
7. If asked for a failure, upload `demo_images/9_failure_basal_cell_carcinoma_test1214.png`: the plaintext model is
   right, the block-encrypted model is wrong. Better that you show it than they find it.

Offline backup: the enlarged pictures in `presentation/` and a screen recording of steps 1-5.

## Likely questions and honest answers

* **Why not just encrypt, send, decrypt, then classify?** Then the server sees the plaintext. Here the
  model never does.
* **Is the block scheme secure?** No, not cryptographically. It gives visual protection: the picture is
  unrecognisable and the key is needed to restore it, but there is no diffusion (one changed pixel
  changes one output pixel), the colour palette is kept up to inversion, and one scrambling is used
  per key.
  The per-pixel scheme scrambles much more but even it has weak diffusion (changing one pixel changes
  only part of the output). Neither replaces a standard cipher like AES.
* **Is the encrypted model really as good as plaintext?** On balanced accuracy and AUC it is comparable
  (53.6% / 54.9% vs 53.3%). It is one run per row and the rarest classes have only 23-29 test images,
  so differences of a few points are within noise; we do not claim it is significantly better.
* **What does the key do for the model?** Variant D feeds an embedding of the key into the positional
  encoding. Compared with the same model without it (variant C) on block-encrypted images: balanced
  accuracy 54.9% vs 53.0%, AUC 0.815 vs 0.839. There is no clear benefit; the differences are within
  noise. The per-pixel rows are in `RESULTS.md`.
* **So what do your custom components add?** Honestly, in these experiments no measurable accuracy
  beyond a plain ViT once the scrambling is learnable. What we can show is that the encryption scheme,
  not the model design, decides whether encrypted-image classification works. Testing the components
  with several seeds and harder data is Phase 2.
* **Is this clinically valid?** No. It is a benchmark study on 28x28 public images. It must not be used
  on patients.
* **What if I upload my own photo?** The models only know this dataset's kind of image; anything else
  gets a meaningless answer.
* **What happens with a different key?** The model was trained with one key; another key gives
  scrambled input the model has never seen, so it has to be retrained for each key.

## Do not claim

* that the encryption is secure, certified or suitable for real patient data;
* that encrypted-image classification is "better than" plaintext (only "comparable here");
* any number that is not in `RESULTS.md`.

## Limitations and Phase 2

* Stronger diffusion with a learnable layout (the central open problem: the more thorough the
  scrambling, the harder it is to learn from).
* Several seeds and confidence intervals; larger images (MedMNIST has 64, 128 and 224 pixel versions);
  other datasets (pathology, blood cells, X-rays).
* A key-management story (per-user keys, and what retraining per key costs).
