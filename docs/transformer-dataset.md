# Transformer dataset

`src/dataset.py` turns task #6 labeled chunks into PyTorch inputs. It uses the
same pinned BERT tokenizer as chunk export; it does not download model weights.

```python
import pandas as pd
from torch.utils.data import DataLoader
from src.chunking import load_tokenizer
from src.dataset import ChunkDataset, split_by_fold

rows = pd.read_parquet("data/interim/labels/train/labeled_chunks.parquet")
folds = pd.read_csv("docs/tuned-baseline-results/folds.csv")
train_rows, validation_rows = split_by_fold(rows, folds, validation_fold=0)
tokenizer = load_tokenizer()
train = ChunkDataset(train_rows, tokenizer)
validation = ChunkDataset(validation_rows, tokenizer)
loader = DataLoader(train, batch_size=8, shuffle=True)
```

Each item contains `input_ids` and `attention_mask` (int64, length 512),
`labels` (float32, length 10), and `label_mask` (bool, length 10).
Label order is `PRIORITY_CATEGORIES`; a true mask means include that entry in
the loss. Pass labels and masks to the eventual loss separately from encoder
inputs. Single-sequence BERT inputs do not require explicit token type IDs.

Special tokens are included, padding is masked, and oversized chunks raise an
error rather than being truncated. Tokenization happens on access. The tokenizer
is copied so caller truncation/padding settings cannot change dataset behavior.
Metadata is available in `dataset.rows`, in the same order as dataset items.
Prediction collection must use an unshuffled loader to retain that ordering.

`split_by_fold` uses saved contract assignments, rejects duplicate/missing fold
assignments and official test rows, and requires nonempty train/validation
partitions. It supports subsets of training contracts for smoke runs.

## Masked loss

`src.loss.masked_bce_with_logits(logits, labels, label_mask)` takes matching
two-dimensional tensors and averages binary cross entropy over entries whose
mask is true. Pass raw logits without applying sigmoid first. Each valid
(chunk, category) pair has equal weight; there are no class weights yet.

```python
from src.loss import masked_bce_with_logits

# logits is the eventual model output with shape (batch_size, 10).
loss = masked_bce_with_logits(logits, batch["labels"], batch["label_mask"])
loss.backward()
```

Masked entries contribute neither loss nor gradient. A completely masked or
empty batch returns differentiable zero. The training loop should skip optimizer
updates on a completely masked batch, because weight decay or optimizer momentum
can change parameters even with zero gradients. Half-precision logits use
float32 loss calculation. All inputs must be on the same device.

## Multi-label model

`src.model.ContractClassifier` uses a BERT encoder, pooled CLS output, dropout,
and a linear head with ten independent logits in `PRIORITY_CATEGORIES` order.
Supplying both labels and label masks computes the masked loss above; omitting
both returns logits for inference. Supplying only one raises an error.

```python
from src.model import ContractClassifier, load_pretrained_model

model = load_pretrained_model()  # Downloads pinned base BERT weights if needed.
batch = next(iter(loader))
output = model(**batch)
output.loss.backward()
probabilities = output.logits.sigmoid()  # Independent category probabilities.
model.save_pretrained("data/interim/models/transformer")
restored = ContractClassifier.from_pretrained("data/interim/models/transformer")
restored.eval()
```

The base weights use the same model and revision as the chunk tokenizer. The
classification head starts randomly initialized and must be trained. Saved
checkpoints preserve category names and order, and incompatible category
mappings are rejected. Use `model.eval()` and `torch.no_grad()` for evaluation.
Model tests use tiny random BERT configurations, including a local base-weight
loading check, so they require no full pretrained weight download. They verify
mechanics rather than predictive accuracy.

Verify without downloading models:

```powershell
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_dataset.py
uv run --locked ruff check --no-cache src/dataset.py tests/test_dataset.py
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_loss.py
uv run --locked ruff check --no-cache src/loss.py tests/test_loss.py
uv run --locked python -B -m pytest -q -p no:cacheprovider tests/test_model.py
uv run --locked ruff check --no-cache src/model.py tests/test_model.py
```
