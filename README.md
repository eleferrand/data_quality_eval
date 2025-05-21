### Data Quality Metrics for Field Linguistics Collections
This repository provides implementations of two metrics designed to evaluate the quality of data in field linguistics collections.

1. PDM Score

The first method calculates the Levenshtein distance between the orthographic transcription and a phonetic transcription. The phonetic transcription is automatically generated using Allosaurus, a universal phoneme recognizer, and both sequences are converted to ASCII format to ensure compatibility.

2. CTC Alignment Score
   
The second method computes the posterior probability of a Connectionist Temporal Classification (CTC) alignment between an orthographic transcription and acoustic features extracted using a pretrained Wav2Vec 2.0 model (trained on English speech). This serves as a proxy for assessing how well the audio matches the transcription.


Dependencies
To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Steps

The first step is to import some data. Your data needs to be a list of dictionaries. Each dict needs to have a key named `ref` that corresponds to the path to your audio file, and `sentence`, which is the transcription. To try it out you can find a small dataset in Seediq in the data folder. 

```
import csv

data = []
with open("data/metadata.csv", "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar="|")
    for row in reader:
        data.append({"ref" : row[0], "sentence" : row[1]})
```

Once your data is loaded, you can calculate either score. You can calculate the PDM score as follows:

```
from eval_align import phone_align

phone_align(data,"Seediq")
```

Or the CTC alignment score as follows:
```
from eval_align import get_ctc_score
get_ctc_score(data, "Seediq):
```

Both functions create an additional key for each data entry named `score` once all the scores are computed you can sort your collection from the lowest to the higher score (higher is better)

```
for ex in sorted(data, key=lambda x : x["score"]):
    print(ex["sentence"], ex["score"])
```

What determines what is a good or bad score depends on your collection. A collection that has a lot of non-ASCII characters or is particularly noisy will have lower scores. You should use these metrics to help you spot major mistakes in your collection.

If you are using either metrics, please cite our work:

```
@inproceedings{le2025that,
  title={That doesn't sound right: Evaluating speech transcription quality in field linguistics corpora},
  author={Le Ferrand, {\'E}ric and Jiang, Bo and Hartshorne, Joshua and Prud'hommeaux, Emily},
  booktitle={Proceedings of the 63th Annual Meeting of the Association for Computational Linguistics},
  year={2025}
}
```
## References

Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.

Li et al. (2020). Universal Phone Recognition with Allosaurus.
