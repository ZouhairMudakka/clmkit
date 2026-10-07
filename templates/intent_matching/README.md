# Support-intent matching and calibrated routing

These two recipes use the complete prepared training galleries and the pinned
model configuration from the evidence protocol. BANKING77 returns examples with
intent labels. A matching label is a relevance proxy, **not a verified resolution,
duplicate ticket, or permission to perform an action**. CLINC150 demonstrates its
own validation-calibrated accept/reject decision; its score is not a probability
and its threshold is not an authorization rule.

Run from a source checkout in the authorized Codespace with clmkit and its HF
dependencies installed. No additional package is required by this template.
Commands below use the evidence workflow's prepared data, protocol, trained
checkpoint, and seal. Preparation downloads public benchmark data only in the
Codespace; source license notices and transformations are recorded in its manifests.
The example reads training-gallery and development files, never held-out queries.

## BANKING77: pinned model, then adaptation and rebuilding

```bash
python templates/intent_matching/intent_matching.py build \
  --dataset banking77 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model minilm --index /workspaces/intent-indexes/banking77-frozen

python templates/intent_matching/intent_matching.py query \
  --dataset banking77 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model minilm --index /workspaces/intent-indexes/banking77-frozen \
  --text "How can I replace my lost card?"
```

`minilm` means the exact model revision and dataset-specific maximum length in
the protocol. To train the registered BANKING77 model, follow the development
workflow rather than choosing a checkpoint from test results:

```bash
python validation/evidence_run.py train \
  --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --seed 42 --output /workspaces/evidence-results/training/42

python templates/intent_matching/intent_matching.py build \
  --dataset banking77 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model checkpoint --checkpoint /workspaces/evidence-results/training/42/final \
  --index /workspaces/intent-indexes/banking77-adapted-42

python templates/intent_matching/intent_matching.py query \
  --dataset banking77 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model checkpoint --checkpoint /workspaces/evidence-results/training/42/final \
  --index /workspaces/intent-indexes/banking77-adapted-42 \
  --text "How can I replace my lost card?"
```

Skip the training command if that run already exists; the evidence runner refuses
to overwrite it. Complete all preregistered seeds for the reported comparison.
Training pairs come only from the gallery. Adaptation may help, hurt, or make no
clear difference; these commands establish a usable workflow, not a quality claim.

Every checkpoint file contributes to its immutable identity. Changed weights,
tokenizer/configuration, protocol, or gallery require a new index directory.
Build embeds the entire gallery, stages a snapshot, strictly reloads it, checks
document IDs/text/labels, then renames that fresh directory. Existing indexes are
preserved. Switch clients to the new path only after success. This is a
single-writer example, not transactional storage; failed staging directories remain
for inspection. Do not mutate an encoder or checkpoint while the example is running.

## CLINC150: separate gallery and calibrated rejection

First finish the full development matrix and create its seal using the evidence
workflow. Use the **CLINC dense** threshold from that exact protocol and model:

```bash
python templates/intent_matching/intent_matching.py build \
  --dataset clinc150 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model minilm --index /workspaces/intent-indexes/clinc150-frozen

python templates/intent_matching/intent_matching.py query \
  --dataset clinc150 --data-root /workspaces/evidence-data \
  --protocol /workspaces/evidence-results/protocol.json \
  --model minilm --index /workspaces/intent-indexes/clinc150-frozen \
  --threshold /workspaces/evidence-results/dev/clinc150/dense/threshold.json \
  --seal /workspaces/evidence-results/seal.json \
  --text "What is my account balance?"
```

The command verifies the threshold's sealed checksum, dataset, full-development
coverage, method, model identity, and protocol/gallery hashes before searching.
Only in-scope CLINC training examples enter the index. Rejected requests return
`intent: null`; candidate examples remain visible for inspection. No operation
is executed. Never apply this threshold to BANKING77 or a different gallery.

Outputs include the full gallery count, immutable encoder identity, matched
document IDs, intent labels, text, and scores. Interpret results using the
matching evidence report and its uncertainty/limitations. Synthetic unit tests
check mechanics and failure handling only; they are not semantic-accuracy evidence.
