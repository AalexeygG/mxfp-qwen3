# MXFP4 and MXFP8 on Qwen3

What the OCP microscaling formats actually cost on a transformer, measured rather than quoted,
and what is left to gain on top of them.

The models are Qwen3-4B and Qwen3-14B. Qwen3-0.6B and Qwen3-1.7B appear in the layer analysis as
two extra size points, so that the claim about what generalizes is checked over a 25x range of
parameter count rather than asserted from a pair.

Everything here is measured on real checkpoints with
[microsoft/microxcaling](https://github.com/microsoft/microxcaling) doing the quantization and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) doing the scoring,
on full task sets with no subsampling.

## The format in one formula

A block of 32 values along the reduction axis shares one 8 bit E8M0 exponent. Each value is
FP4 (E2M1) or FP8 (E4M3). So the storage cost per value is

    bits per value = element bits + 8 / 32

    MXFP4: 4 + 0.25 = 4.25 bits, 3.765x smaller than bf16
    MXFP8: 8 + 0.25 = 8.25 bits, 1.939x smaller than bf16

The block scale is a quarter bit per value. On MXFP4 that is six percent of the budget, which is
why "four times smaller" is wrong and 3.765x is right.

## Measured footprint

All 2D tensors quantized, norms left in bf16, tied embeddings counted once.

| model | bf16 | MXFP8 | MXFP4 | ratio |
|---|---|---|---|---|
| Qwen3-0.6B | 1 192 099 840 B, 1.11 GiB | 0.573 GiB | 0.295 GiB | 3.764 |
| Qwen3-1.7B | 3 441 149 952 B, 3.21 GiB | 1.653 GiB | 0.851 GiB | 3.764 |
| Qwen3-4B | 8 044 936 192 B, 7.49 GiB | 3.863 GiB | 1.990 GiB | 3.764 |
| Qwen3-14B | 29 536 614 400 B, 27.51 GiB | 14.184 GiB | 7.307 GiB | 3.764 |

The ratio is not exactly 3.765 because the one dimensional tensors stay in bf16. They cost almost
nothing, 65 KB on Qwen3-0.6B, but leaving them out of the accounting would be dishonest.

Two things are easy to get wrong here. Qwen3-0.6B ties `lm_head` to `embed_tokens`, and the
checkpoint stores both copies, so a naive sum reports 751M parameters instead of the real 596M.
And microxcaling does fake quantization, it returns bf16 tensors that happen to sit on the MXFP4
grid, so nothing shrinks at runtime. Nothing here is read off a memory profiler.

`scripts/verify_packing.py` writes the packed codes and exponents to a real file so the number is
not just arithmetic. On Qwen3-0.6B, 595 984 384 values in 18 624 512 blocks:

| format | file on disk | formula | difference | measured |
|---|---|---|---|---|
| MXFP4 | 316 616 704 B | 316 616 704 B | 0 | 4.2500 bits per value |
| MXFP8 | 614 608 896 B | 614 608 896 B | 0 | 8.2500 bits per value |

Unpacking the file reproduces the quantized tensors exactly, maximum round trip error 0. The
131 072 byte gap against the table above is the one dimensional tensors, which stay in bf16 and
are not packed.

## Quality

Two measurements, because they do not agree, and the disagreement is the interesting part.

Perplexity on wikitext2, measured directly. The cost of MXFP4 on the weights, across the whole
size range:

| model | bf16 | MXFP8 weights | MXFP4 weights | MXFP4 cost |
|---|---|---|---|---|
| Qwen3-0.6B | 21.32 | 21.48 | 25.58 | +20.0% |
| Qwen3-1.7B | 16.85 | 17.59 | 21.73 | +28.9% |
| Qwen3-4B | 15.17 | 15.41 | 16.48 | +8.7% |
| Qwen3-14B | 10.35 | 10.40 | 11.15 | +7.7% |

Larger models take MXFP4 better, and the effect is large: 20% on the smallest against 7.7% on the
largest. On perplexity it is not monotone, Qwen3-1.7B is the worst of the four, so on that metric
it is a trend rather than a law. On the benchmark it is monotone and clean:

| model | arc_easy bf16 | arc_easy MXFP4 | loss |
|---|---|---|---|
| Qwen3-0.6B | 0.6082 | 0.5362 | 7.2 points |
| Qwen3-4B | 0.8047 | 0.7811 | 2.4 points |
| Qwen3-14B | 0.8418 | 0.8279 | 1.4 points |

Which is the practical form of the statement: MXFP4 gets cheaper exactly where the memory saving
matters most. MXFP8 costs between 0.4% and 4.4% and is the format to reach for when the margin matters.

Both operands of the matmul, on the two models where the full matrix was run:

| configuration | Qwen3-0.6B | Qwen3-4B | Qwen3-14B |
|---|---|---|---|
| bf16 | 21.32 | 15.17 | 10.35 |
| weights MXFP8 | 21.48 | 15.41 | 10.40 |
| activations MXFP8 | 21.47 | 15.53 | |
| weights and activations MXFP8 | 21.62 | 15.71 | 10.46 |
| weights MXFP4 | 25.58 | 16.48 | 11.15 |
| activations MXFP4 | 25.74 | 17.70 | |
| weights MXFP4, activations MXFP8 | 25.91 | 16.76 | 11.23 |

MXFP8 is close to free on both operands at every size. On the small model MXFP4 costs the same
whether applied to weights or activations, and the two costs add rather than multiply.

lm-evaluation-harness, full task sets, no subsampling. Qwen3-0.6B:

| configuration | arc_easy | piqa | winogrande | lambada ppl | lambada acc |
|---|---|---|---|---|---|
| bf16 | 0.6082 | 0.6730 | 0.5572 | 24.71 | 0.3998 |
| weights MXFP8 | 0.6115 | 0.6714 | 0.5762 | 23.20 | 0.4075 |
| weights and activations MXFP8 | 0.6136 | 0.6643 | 0.5525 | 23.39 | 0.4087 |
| weights MXFP4 | 0.5362 | 0.6485 | 0.5643 | 39.26 | 0.3522 |
| weights MXFP4 plus our method | 0.5109 | 0.6491 | 0.5462 | 31.90 | 0.3714 |

Standard errors are 0.010 on arc_easy, 0.011 on piqa, 0.014 on winogrande and 0.007 on lambada
accuracy, so differences below about two points are not differences. On that basis MXFP8 is
indistinguishable from bf16 on every task, and that holds with both operands of the matmul
quantized, not just the weights. MXFP4 costs 7.2 points of arc_easy and 59% of lambada
perplexity. Winogrande does not separate anything here: MXFP4 scores above bf16 on it, which is
noise, not an improvement.

## What the layer analysis says

The headline is a negative one. Across 929 tensors of Qwen3-0.6B, 1.7B, 4B and 14B, MXFP4 signal
to quantization noise sits between 18.58 and 18.71 dB on average per model, with a standard
deviation of 0.06 to 0.10 dB inside each. Twenty five times more parameters moves it by 0.12 dB.
Layer depth and projection type move it by less than that.

| model | parameters | MXFP4 SQNR | MXFP8 SQNR | block kurtosis |
|---|---|---|---|---|
| Qwen3-0.6B | 596 M | 18.667 dB | 30.428 dB | 2.99 |
| Qwen3-1.7B | 1.72 B | 18.707 dB | 30.542 dB | 3.02 |
| Qwen3-4B | 4.02 B | 18.633 dB | 30.324 dB | 2.96 |
| Qwen3-14B | 14.77 B | 18.583 dB | 30.275 dB | 3.10 |

That kills the usual assumption that some layers compress better and deserve fewer bits. Under
block scaling they do not.

The reason shows up in which statistic predicts the error. Correlation with SQNR over all tensors:

| statistic | correlation with SQNR, 929 tensors |
|---|---|
| kurtosis of the whole tensor | -0.10 |
| kurtosis inside a 32 value block | -0.44 |
| block crest factor | -0.54 |
| fraction of blocks with a saturated maximum | -0.52 |

The outlier statistic that motivates AWQ and SmoothQuant has no predictive power for MX, because
a per-block scale gives every outlier its own exponent. What matters is the shape inside the
block, and that shape is Gaussian everywhere: block kurtosis sits at 3.0 across every projection
type in all four models.

The structure that does exist is ordered by exactly that statistic. Mean MXFP4 SQNR by
projection, pooled over all four models:

| role | SQNR | block kurtosis |
|---|---|---|
| down_proj | 18.571 dB | 3.141 |
| v_proj | 18.627 dB | 3.069 |
| k_proj | 18.627 dB | 2.984 |
| gate_proj | 18.650 dB | 3.109 |
| up_proj | 18.658 dB | 2.954 |
| q_proj | 18.659 dB | 2.999 |
| o_proj | 18.687 dB | 2.874 |

`down_proj` is the worst everywhere and `o_proj` the best, and the ordering follows block
kurtosis rather than anything about what the layer does. The gap between down_proj and the rest
is 0.08 dB pooled, 0.09 to 0.11 dB on three of the four models and nil on Qwen3-1.7B.

The deviation concentrates in early layers. For down_proj in Qwen3-14B, the first quarter of the
stack has block kurtosis 4.04 and SQNR 18.34 dB, against 2.91 and 18.56 dB in the last quarter.
So the exceptions do not break the mechanism, they are the mechanism: where the within block
distribution stops being Gaussian, the error follows.

Quantizing synthetic data confirms the mechanism end to end.

| source | kurtosis | MXFP4 SQNR |
|---|---|---|
| gaussian | 3.0 | 18.78 dB |
| laplace | 6.0 | 17.95 dB |
| student t, 3 dof | 121 | 16.98 dB |
| gaussian with 1% outliers at 30 sigma | 244 | 15.68 dB |

MX error does depend on distribution shape in general. Transformer weights simply sit at the
Gaussian point, which is why 18.7 dB shows up everywhere. That is the useful, transferable part:
MXFP4 weight error on a transformer is predictable from theory without touching the checkpoint.

One useful side check. The per tensor statistics were computed twice, once on an Apple M3 with
the pure torch path and once on a Colab T4 with CUDA available, and the two agree exactly: mean
MXFP4 SQNR 18.6668 dB against 18.6668 dB on Qwen3-0.6B, and the same to four decimals on 1.7B and
4B. The quantization path has no hardware dependent behaviour, which is worth knowing before
trusting numbers measured on one machine.

## Compression on top of the format

Measured per tensor with zstd, comparing against the 4.25 and 8.25 bit baselines.

| stream | naive cost | after coding | gain |
|---|---|---|---|
| MXFP4 codes | 4.0 bits | 3.85 bits | 1.04x |
| MXFP8 codes | 8.0 bits | 6.67 bits | 1.20x |
| E8M0 scales | 0.25 bits | 0.041 bits | 6.1x |

The scale stream is where the free win is. Neighbouring blocks in a row have almost the same
order of magnitude, so delta coding the exponents drops their entropy from 8 bits to about 1.2.
Lossless, no quality change at all: MXFP4 goes from 4.25 to 3.89 bits per value and MXFP8 from
8.25 to 6.71.

The codes themselves are a different story. MXFP4 code entropy is 3.81 bits out of 4 and zstd
reaches 3.86, so the coder is already at its own bound. Two context models were measured to see
whether there is structure an order zero coder misses, and there is, but not enough to pay for
itself:

| model | entropy, MXFP4 | gain | cost of signalling the context |
|---|---|---|---|
| order zero, what we use | 3.813 bits | | |
| conditioned on block class | 3.768 bits | 0.045 | 0.073 bits per value |
| block maximum coded separately | 3.708 bits | 0.105 | 0.156 bits per value |

Both lose more to telling the decoder which model to use than they gain. Capturing that 0.1 bits
would need an adaptive arithmetic coder that derives the context from the already decoded part of
the block, which is a different piece of engineering and out of scope here. The practical
statement is that this compression sits within about three percent of what is reachable without
one.

Where that leaves the footprint, measured over the tensors of two models:

| format | as specified | after coding | vs bf16 |
|---|---|---|---|
| MXFP4 | 4.25 bits | 3.896 bits | 3.765x becomes 4.107x |
| MXFP8 | 8.25 bits | 6.650 bits | 1.939x becomes 2.406x |

Both are lossless, and that is tested rather than asserted. `scripts/test_roundtrip.py` compresses
every tensor of a model, decompresses it, and compares:

| model | format | tensors | mismatches | bits per value | vs bf16 |
|---|---|---|---|---|---|
| Qwen3-0.6B | MXFP4 | 197 | 0 | 3.905 | 4.097x |
| Qwen3-0.6B | MXFP8 | 197 | 0 | 6.648 | 2.407x |
| Qwen3-4B | MXFP4 | 253 | 0 | 3.945 | 4.056x |
| Qwen3-4B | MXFP8 | 253 | 0 | 6.641 | 2.409x |

The test earned its place: it found that the delta coding of the exponents was dropping the first
value of every row, so the decoder came back with everything shifted by a constant. The sizes
above are after that fix.

## Negative results

Three ideas were implemented, measured and rejected. They are kept here because the measurements
are the point.

**Coarsening codes toward more probable neighbours.** This was the starting idea, taken from an
earlier project: do not zero anything, just nudge unimportant values to a neighbouring level and
let the entropy coder profit.  Rate distortion optimal, one level of
movement, importance weighted. On MXFP4 it moves 1.3% of codes and saves 0.005 bits per value,
because the alphabet has only 16 symbols and is already near uniform. On MXFP8 it saves about 4%
at a visible SQNR cost. Not worth it either way.

**Bit allocation across layers by absolute sensitivity.** Choosing MXFP4 or MXFP8 per tensor to
minimise the activation weighted squared error under a bit budget looked excellent on the proxy,
5.4x less predicted error at 4.12 bits per value. Perplexity got worse, 36.7 against 25.6 for
plain MXFP4. The objective was wrong: absolute error is dominated by tensors with large weights
and large activations, and the rank correlation between absolute and relative error across
tensors is only 0.20. Once normalised, relative sensitivity varies by a factor of 1.8 across all
196 tensors, which is consistent with the SQNR result and means there is very little to allocate.

**Activation weighted exponent choice.** Picking the block exponent that minimises activation
weighted error made perplexity worse, 26.04 against 25.58, while the unweighted version helped
slightly. Protecting high activation channels at the expense of everything else does not pay off
at block granularity.

## What partly works

Two changes, neither of which costs a single extra bit or breaks format compliance. Both help,
but only on one of the two kinds of metric, and that split is the main result of this section.

**Choosing the block exponent instead of deriving it.** The saturation row in the table above
points at a real defect. The standard rule sets the exponent to `floor(log2(max)) - emax`, which
puts the block maximum in [4, 8) while FP4 represents at most 6. In 41% of MXFP4 blocks the
largest element is clipped by up to 25%. Picking the exponent per block by minimising the actual
block error recovers 0.30 dB on MXFP4 and 1.21 dB on MXFP8. The scale is still one 8 bit E8M0
value, so the stored format is unchanged.

**Compensating the quantization error through the rest of the weight.** This follows the GPTQ
construction of Frantar et al., arXiv 2210.17323, applied to MX blocks. Quantizing a block leaves
a residual, and the remaining columns of the same row can absorb part of it if you know how the
layer input is distributed. With a Hessian from 16 calibration sequences and a group size equal
to the MX block, the residual of each block is projected onto the columns not yet quantized.
Layers are processed in order, so each group is calibrated through the already quantized ones.

On perplexity this works well, and the two parts can be separated. Qwen3-4B, wikitext2, 32
windows of 1024 tokens, float16 on a T4:

| configuration | perplexity | gap to bf16 | share of the gap removed | bits per value |
|---|---|---|---|---|
| bf16 | 15.1675 | | | 16 |
| MXFP4, standard | 16.4830 | 1.3155 | | 4.25 |
| MXFP4, exponent search only | 16.3503 | 1.1828 | 10.1% | 4.25 |
| MXFP4, error compensation only | 15.7037 | 0.5362 | 59.2% | 4.25 |
| MXFP4, both | 15.5000 | 0.3325 | 74.7% | 4.25 |

Each part works on its own and the two are better together than the sum of their separate
contributions. Qwen3-0.6B gives the same ordering with a smaller total: 21.32 bf16, 25.58 plain
MXFP4, 25.44 with the exponent search, 23.32 with compensation, 22.91 with both, which is 63% of
the gap. The method helps more on the larger model.

On accuracy the answer depends on the model, and that is the interesting part.

Qwen3-4B under lm-evaluation-harness, full task sets:

| configuration | arc_easy | piqa | winogrande | lambada ppl | lambada acc |
|---|---|---|---|---|---|
| bf16 | 0.8047 | 0.7508 | 0.6598 | 7.2896 | 0.5969 |
| weights MXFP8 | 0.8009 | 0.7497 | 0.6480 | 7.1643 | 0.6018 |
| weights and activations MXFP8 | 0.7992 | 0.7470 | 0.6472 | 7.2406 | 0.5993 |
| weights MXFP4 | 0.7811 | 0.7356 | 0.6322 | 8.3163 | 0.5756 |
| weights MXFP4 plus our method | 0.7984 | 0.7307 | 0.6559 | 7.4312 | 0.5931 |

The share of the MXFP4 loss the method gives back: 73% of arc_easy, 86% of winogrande, 82% of
lambada accuracy, 86% of lambada perplexity. What is left of the gap to bf16 is 0.8 standard
errors on arc_easy, 0.3 on winogrande, 0.6 on lambada, which is to say the quantized model is no
longer distinguishable from the original on three of the four. Piqa is the exception, it does not
improve and sits 1.9 standard errors below bf16.

On Qwen3-0.6B the same configuration behaves differently: it improves lambada perplexity from
39.26 to 31.90 and lambada accuracy from 0.3522 to 0.3714, but arc_easy drops from 0.5362 to
0.5109, about 2.5 standard errors below plain MXFP4. Swapping the wikitext calibration for a
slice of C4 web text recovers part of it, arc_easy 0.5194 and lambada perplexity 30.00, but not
all.

So the regression is a property of the small model, not of the method. At 4B the method helps
every metric that MXFP4 damaged. The honest way to state it is that error compensation needs the
model to have enough capacity for the correction to be worth making, and on a 0.6B model fitting
the calibration set more tightly can cost more than it returns.

Both operands quantized to MXFP8 is indistinguishable from bf16 on Qwen3-4B as well, at most 0.9
standard errors on any task, which makes it the safe default when the memory has to come down and
the quality must not move.

## Layout

    src/mxfmt.py        MX quantization, code and scale extraction, exponent search
    src/stats.py        per tensor error and distribution statistics
    src/compress.py     rate distortion coarsening and entropy coding
    src/errcomp.py         error compensated quantization with MX sized groups
    src/fakequant.py    in place quantization of a loaded model, weights and activations
    src/modelio.py      streaming access to safetensors shards
    scripts/verify_packing.py   writes packed weights to disk and checks the size
    scripts/test_roundtrip.py   compresses and decompresses every tensor and compares
    scripts/            one entry point per stage, see below
    colab/run.ipynb     self contained notebook for a single GPU runtime
    kaggle/run.ipynb    same, split over two T4 so that Qwen3-14B fits

## Setup

    python -m venv venv && ./venv/bin/pip install -r requirements.txt
    git clone --depth 1 https://github.com/microsoft/microxcaling.git vendor/microxcaling

The quantization itself comes from microxcaling, this project does not reimplement it. What is
added on top is the extraction of codes and scales for analysis, the storage accounting, the
entropy coding, and the two changes in the section above.

## Running it

    python scripts/quantize_model.py Qwen/Qwen3-0.6B     # per tensor statistics
    python scripts/memory_report.py                      # footprint table
    python scripts/verify_packing.py Qwen/Qwen3-0.6B     # pack to disk, check against the formula
    python scripts/test_roundtrip.py Qwen/Qwen3-0.6B     # compress, decompress, compare
    python scripts/calibrate.py Qwen/Qwen3-0.6B          # activation norms
    python scripts/sensitivity.py Qwen/Qwen3-0.6B        # cost and error per config
    python scripts/allocate.py Qwen/Qwen3-0.6B           # bit budget allocation
    python scripts/ppl.py Qwen/Qwen3-0.6B --weights mxfp4
    python scripts/run_eval.py Qwen/Qwen3-0.6B --tasks arc_easy,piqa --weights mxfp4
    python scripts/analyze.py                            # tables and figures

The statistics stage streams shards one tensor at a time, so model size is bounded by disk rather
than by RAM, and Qwen3-14B was analysed that way on a laptop with 8 GB.

Evaluation is the part that needs the model resident. Qwen3-4B is 8 GB of weights and fits one
T4. Qwen3-14B is 27.5 GB and does not, so it is split across the two T4 of a Kaggle session with
`--device-map auto`, 32 GB in total. T4 is Turing and has no usable bf16, so those runs are in
float16, which is stated wherever their numbers appear.
