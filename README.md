README.md# Gloss-Based Pipeline for Spoken to Signed Language Translation

a `text-to-gloss-to-pose-to-video` pipeline for spoken to signed language translation.

- Demos available for:
  - 🇩🇪 [Swiss German Sign Language](https://sign.mt/?sil=sgg&spl=de) 🇨🇭
  - 🇫🇷 [French Sign Language of Switzerland](https://sign.mt/?sil=ssr&spl=fr)🇨🇭
  - 🇮🇹 [Italian Sign Language of Switzerland](https://sign.mt/?sil=slf&spl=it) 🇨🇭

- Paper available on [arxiv](https://arxiv.org/abs/2305.17714), presented
  at [AT4SSL 2023](https://sites.google.com/tilburguniversity.edu/at4ssl2023/).

![Visualization of our pipeline](assets/pipeline.jpg)

## Install

```bash
pip install git+https://github.com/ZurichNLP/spoken-to-signed-translation.git
```

## Usage

For language codes, we use the [IANA Language Subtag Registry](https://www.iana.org/assignments/language-subtag-registry/language-subtag-registry).
Our pipeline provides multiple scripts. 

To quickly demo it using a dummy lexicon, run:

<a target="_blank" href="https://colab.research.google.com/drive/1UtBmfBIhUa2EdLMnWJr0hxAOZelQ50_9?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```bash
git clone https://github.com/ZurichNLP/spoken-to-signed-translation
cd spoken-to-signed-translation

text_to_gloss_to_pose \
  --text "Kleine Kinder essen Pizza in Zürich." \
  --glosser "simple" \
  --lexicon "assets/dummy_lexicon" \
  --spoken-language "de" \
  --signed-language "sgg" \
  --pose "quick_test.pose"
```



#### Text-to-Gloss Translation

This script translates input text into gloss notation. 

```bash
text_to_gloss \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf>
```

#### Pose-to-Video Conversion

This script converts a pose file into a video file.

```bash
pose_to_video \
  --pose <pose_file_path>.pose \
  --video <output_video_file_path>.mp4
```

#### Text-to-Gloss-to-Pose Translation

This script translates input text into gloss notation, then converts the glosses into a pose file.

```bash
text_to_gloss_to_pose \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --pose <output_pose_file_path>.pose
```

Add `--coverage-info` to print per-token lexicon coverage to stdout, or `--coverage-stats <file.json>` to save it to a JSON file.

#### Bulk Text-to-Gloss-to-Pose Translation

This script processes a file of texts (one sentence per line) and writes one pose file per line to an output directory.

```bash
text_to_gloss_to_pose_bulk \
  --texts <input_texts_file> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --output-dir <output_directory> \
  --coverage-stats <coverage_file>.json
```

The `--coverage-stats` argument is optional. When provided, per-token coverage statistics are accumulated across all sentences and saved as a JSON file. The JSON includes the overall fraction of matched tokens and a per-sentence, per-token breakdown of which glosses were found in the lexicon and which were not.

##### Compacted output (chunked poses)

By default each sentence produces its own `.pose` file (`000000.pose`, `000001.pose`, …). When working with large datasets this results in many small files. Use `--compacted-poses` to concatenate the generated poses into larger chunk files instead:

```bash
text_to_gloss_to_pose_bulk \
  --texts <input_texts_file> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --output-dir <output_directory> \
  --compacted-poses \
  --max-frames-per-chunk 10000
```

The output directory will contain:

- **Chunk pose files** — `chunk_000000.pose`, `chunk_000001.pose`, …
  A new chunk is started whenever adding the next sentence's pose would exceed `--max-frames-per-chunk` frames (default: `10000`). A single pose that already exceeds the limit is written as its own chunk.

- **`metadata.tsv`** — a tab-separated file with one row per input sentence, so you can locate any sentence's pose within its chunk:

  | Column | Description |
  |---|---|
  | `text` | The source sentence |
  | `pose_file` | Absolute path to the chunk `.pose` file |
  | `start_frame` | Index of the first frame belonging to this sentence within the chunk |
  | `end_frame` | Index of the last frame (inclusive) belonging to this sentence within the chunk |

- **`<coverage_file>.json`** — if `--coverage-stats` is also provided, coverage is still accumulated and saved as usual.

#### Visualizing Coverage

The `scripts/visualize_coverage.py` script renders a coverage JSON file (produced by `--coverage-stats`) in the terminal, coloring each gloss token by how it was matched:

```bash
python scripts/visualize_coverage.py <coverage_file>.json
```

Example output:

```
Legend:
  ■ matched via lexicon
  ■ matched via language backup
  ■ matched via fingerspelling
  ■ not matched

Sentence: Kleine Kinder essen Pizza in Zürich
Gloss:    KLEIN KIND ESSEN PIZZA IN ZÜRICH

Overall coverage: 0.833 (5/6 tokens matched)
```

| Color | Meaning |
|---|---|
| Green | Matched directly in the lexicon |
| Yellow | Matched via a language backup (e.g. `slf→ise`) |
| Orange | Matched via fingerspelling |
| Red | Not matched at all |

#### Text-to-Gloss-to-Pose-to-Video Translation

This script translates input text into gloss notation, converts the glosses into a pose file, and then transforms the pose file into a video.

```bash
text_to_gloss_to_pose_to_video \
  --text <input_text> \
  --glosser <simple|spacylemma|rules|nmt> \
  --lexicon <path_to_directory> \
  --spoken-language <de|fr|it> \
  --signed-language <sgg|ssr|slf> \
  --video <output_video_file_path>.mp4
```

## Methodology

The pipeline consists of three main components:

1. **Text-to-Gloss Translation:**
   Transforms the input (spoken language) text into a sequence of glosses.

- [Simple lemmatizer](src/text_to_gloss/simple.py),
- [Spacy lemmatizer: more accurate, but slower lemmatization, covering fewer languages than `simple`](src/text_to_gloss/spacylemma.py),
- [Rule-based word reordering and dropping](src/text_to_gloss/rules.py) component
- [Neural machine translation system](src/text_to_gloss/nmt.py).

2. **Gloss-to-Pose Conversion:**

- [Lookup](src/gloss_to_pose/lookup.py): Uses a lexicon of signed languages to convert the sequence of glosses into a
  sequence of poses.
- [Pose Concatenation](src/gloss_to_pose/concatenate.py): The poses are then cropped, concatenated, and smoothed,
  creating a pose representation for the input sentence.

3. **Pose-to-Video Generation:** Transforms the processed pose video back into a synthesized video using an image
   translation model.

## Supported Languages

| Language                    | IANA Code | Glossers Supported                                                                                                                                         |
|-----------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Swiss German Sign Language  | sgg       | `simple`, `spacylemma`, `rules`, [`nmt`](https://github.com/ZurichNLP/spoken-to-signed-translation/tree/main/spoken_to_signed/text_to_gloss#nmt-component) |
| Swiss French Sign Language  | ssr       | `simple`, `spacylemma`                                                                                                                                     |
| Swiss Italian Sign Language | slf       | `simple`, `spacylemma`                                                                                                                                     |
| German Sign Language        | gsg       | `simple`, `spacylemma`, [`nmt`](https://github.com/ZurichNLP/spoken-to-signed-translation/tree/main/spoken_to_signed/text_to_gloss#nmt-component)          |
| British Sign Language       | bfi       | `simple`, `spacylemma`, [`nmt`](TODO-model-link)                                                                                                           |


## Citation

If you find this work useful, please cite our paper:

```bib
@inproceedings{moryossef2023baseline,
  title={An Open-Source Gloss-Based Baseline for Spoken to Signed Language Translation},
  author={Moryossef, Amit and M{\"u}ller, Mathias and G{\"o}hring, Anne and Jiang, Zifan and Goldberg, Yoav and Ebling, Sarah},
  booktitle={2nd International Workshop on Automatic Translation for Signed and Spoken Languages (AT4SSL)},
  year={2023},
  month={June},
  url={https://github.com/ZurichNLP/spoken-to-signed-translation},
  note={Available at: \url{https://arxiv.org/abs/2305.17714}}
}
```
