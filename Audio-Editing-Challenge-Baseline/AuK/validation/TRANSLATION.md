# Reference-voice speech translation in PE

## Flow and compatibility

Classification stays `content_edit/replace`, with `edit_intent=translation`,
`translation_scope=full|partial`, and target language `zh|en` from the user/MMAE
instruction. No new task/subtype or Router/public API is introduced.

`prepare()` transcribes the reference audio, classifies the request, then invokes
`_prepare_translation()` BEFORE the unchanged `orig/new` required-slot checks.
The source language is the ASR language label, not the user's instruction language.
An optional explicit source-language hint is used only to reject conflicts.
The existing configured LLM translates ASR text; the generated `orig/new` slots and
source/target/instruction languages are retained in PE parameters/debug output.

For full speech translation, `_render_instruction()` uses the configurable
`content_edit.full_translation_template`:

```
Say the following with the same voice: "{translated text}"
```

The entire original reference is preserved: no VAD trimming, no audio replacement
with a transcript. `ref_text` contains the transcript, `gen_text` contains the translation,
and `generation_mode=reference_tts` records the backend strategy. The existing AuK server
passes this instruction and reference to `AukInfer.generate()` unchanged.

Normal content edits retain the old templates and content-scaled duration and make no
translation call. Partial translations select a unique exact ASR substring or a 1-based
punctuation-delimited sentence index and still render a replacement instruction. Missing,
repeated or conflicting selectors are rejected. Time/speaker/language/event selection
requires reliable segmentation/alignment and is not implemented. Character offsets in
`source_span` are NOT audio timestamps. ASR punctuation is not guaranteed to match actual
sentence boundaries. AuK generation does not guarantee sample-exact preservation elsewhere.

Full translation estimates duration from target text using the existing text-duration
helpers; it does not apply the ordinary source-content replacement ratio. Partial translation
retains a bounded source/target text-unit ratio. A positive explicit `gen_seconds` takes
precedence. All automatic duration estimates are heuristic, not a validated optimum.

## Language and capability guards

`_sensevoice_language()` preserves zh/en/yue/ja/ko/nospeech; missing tags remain unknown.
The existing zh/en instruction/template normalization is unchanged. Blank/error/punctuation-only
ASR results, unknown source labels, unsupported targets and source-hint mismatches stop translation.
Fallback ASR skips error-bearing transcripts even when nonempty.

SenseVoiceSmall's documented source set is zh/en/yue/ja/ko, not Spanish. This does not prove
per-sample correctness: the earlier real Korean MMAE check returned an English label/text.
Explicit Spanish requests are rejected; no new ASR model is installed. If ASR mislabels
unsupported audio and no source hint exists, this component cannot independently discover
all such mistakes. A multilingual provider and reliable language/confidence evidence remain
future work; do not translate guesses just because a transcript is nonempty.

Song translation remains unsupported; explicit lyric replacement remains `vocal_edit`.
Northeast/Taiwanese-accented Mandarin standardization stays `accent_edit`, not translation.
Shanghainese semantic translation needs validated ASR; ambiguous `Taiwanese language`
requests need clarification. These constraints are classifier instructions, not a guarantee
that an untested live LLM will always classify every ambiguous request correctly.

## Validation and evidence

Run from repository root:

```bash
Audio-Editing-Challenge-Baseline/AuK/.venv/bin/python -m unittest discover \
  -s Audio-Editing-Challenge-Baseline/AuK/validation -p test_pe_translation.py -v
```

25 offline tests mock ASR and LLM responses. They cover en→zh, zh→en, ko→en parameter flow,
Spanish rejection, missing/unknown languages, full-vs-partial rendering, complete-reference
preservation, ordinary dog→cat regression, required slots, selectors, fallback and duration.
They do NOT validate actual LLM classification, semantic translation, voice similarity or GPU
speech quality. Classification of songs/dialects is checked as a prompt contract.

Prior direct GPU tests used the same official TTS prompt with English reference audio and
Chinese text. Short Chinese transcription matched; the user reported that the full translation
was successful by listening. SenseVoice returned only punctuation on that long output, so that
ASR result is not proof that generation failed. Do not conflate an ASR verification failure with
a synthesis failure, or correct Chinese content with preserved speaker identity/background.

This code revision has not been tested with fresh real translation API requests: sourcing
`baseline-local-env.sh` in the tool shell provided neither API keys nor endpoint configuration.
Restart AuK before testing; use a new runner output directory because existing WAVs are skipped.
Inspect tool errors and listen to outputs: the existing runner may count original-audio fallback
copies as generated. Validate ASR, translation, synthesized content and voice/background
preservation separately. Continue with en↔zh full speech translation first, then independently
validate Korean, multilingual ASR, selective translation, songs and dialect conversion.
