# Builtin voices

Shared cards. `voice_id` is a role slug, not an engine name — no `fr3-` / `br2-`.
On boot the engine wipes every premade card in `--voice-dir`, then plants
the current union: model-native voices (if the checkpoint has any) plus this
shared pack when the backend sets `uses_shared_pack`. Cloned and designed
cards stay. Premade is not user-owned — users cannot add, edit, or delete
those rows — so a catalog change is visible on the next restart.

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
| `instruction` | FireRed design speak; Breeze Voice Direction |
| `sample_text` | first-speak freeze when there is no wav yet; match instruction language |
| `description` | `GET /v1/voices` card (one string, Chinese then English) |
| `labels` | optional tags (`language`, `accent`, `gender`) |
| `plan` | FireRed: saved 12-item voice plan |

`name` is bilingual (`中文短名 / English Short`). `description` is one
string with a Chinese sentence then an English sentence — the list API has
no `zh` / `en` pair. `instruction` stays the prompt the model consumes and
is not returned on `GET /v1/voices`.

This pack is Chinese and English only. Mandarin and English use character
roles (not gender-paired). Chinese dialects are male+female pairs.

There is no portable speaker embedding. A card is the instruction and/or
the reference wav+transcript the model actually consumes.
