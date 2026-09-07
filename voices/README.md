# Builtin voices

Shared cards. `voice_id` is a role slug, not an engine name — no `fr3-` / `br2-`.
On boot the engine reconciles premade cards in `--voice-dir` with the
current union: model-native voices (if the checkpoint has any) plus this
shared pack when the backend sets `uses_shared_pack`. A premade id only in
the store is deleted; one only in the union is planted.

Premade has two kinds (`source` on the card, or inferred from a factory wav):

| `source` | Identity (wipe + replant) | Patch only |
|---|---|---|
| `design` | `instruction` | name / description / sample_text / labels |
| `clone` | `prompt.wav` (sha256) + `prompt.txt` | name / description / instruction (Voice Direction) / labels |

A clone card ships `prompt.wav` + `prompt.txt` (exact transcript). Speak uses
that pair; it will not fall back to design if the wav is missing. A design
card still freezes a sample on first speak. Changing a clone card's
instruction does **not** drop the factory wav.

User `cloned` / `generated` / `pending` dirs stay. Premade is not user-owned.

```
<voice_id>/
  meta.json      required
  prompt.wav     optional; clone / preview sample
  prompt.txt     optional; exact transcript of prompt.wav
```

`meta.json` fields:

| field | who needs it |
|---|---|
| `voice_id`, `name`, `category` (`premade`) | list / speak |
| `source` | `design` (instruction freeze) or `clone` (factory wav + transcript) |
| `instruction` | design identity; clone Voice Direction (optional) |
| `sample_text` | design first-speak text when there is no wav yet |
| `description` | `GET /v1/voices` card (one string, Chinese then English) |
| `labels` | optional tags (`language`, `accent`, `gender`) |
| `plan` | FireRed: saved 12-item voice plan |

`name` is bilingual (`中文短名 / English Short`). `description` is one
string with a Chinese sentence then an English sentence — the list API has
no `zh` / `en` pair. `instruction` stays the prompt the model consumes and
is not returned on `GET /v1/voices`.

This pack is four defaults: Mandarin female / male, English female / male
(`zh-f`, `zh-m`, `en-f`, `en-m`). Dialects and character roles are not
shipped as premade; use Voice Design or clone.

There is no portable speaker embedding. A card is the instruction and/or
the reference wav+transcript the model actually consumes.
