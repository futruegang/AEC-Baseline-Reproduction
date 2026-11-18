"""Offline contract tests: mock ASR and API, never claim acoustic/model accuracy."""
import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from auk.infer import pe


def call(stage, payload):
    return pe.LLMCall(stage, 'test', 'test', json.dumps(payload), None, None, {})


class TranslationTests(unittest.TestCase):
    def engine(self, source='The dog runs.', language='en', target='zh', translated='狗在跑。', **extra):
        params = dict(edit_intent='translation', translation_scope='full', target_language=target)
        params.update(extra)
        engine = pe.PromptEnhancer.__new__(pe.PromptEnhancer)
        engine._transcribe = Mock(return_value=pe.ASRCall('SenseVoiceSmall', source, language, {}))
        engine._call_llm = Mock(side_effect=[call('classify', {
            'supported': True, 'task_type': 'content_edit', 'operation_subtype': 'replace',
            'params_extracted': params, 'language': 'en',
        }), call('translate_content', {'translation': translated})])
        return engine

    def prepare(self, engine, **kwargs):
        with tempfile.NamedTemporaryFile(suffix='.wav') as f, \
                patch.object(pe, '_audio_duration', return_value=4.0), \
                patch.object(pe, '_vad_speech_bounds', return_value=None), \
                patch.object(pe, '_prepare_audio', side_effect=lambda p, **kw: (p, [])):
            return engine.prepare('test instruction', f.name, **kwargs)

    def test_full_translation_pairs(self):
        for source, language, target, translated in [
            ('The dog runs.', 'en', 'zh', '狗在跑。'),
            ('狗在跑。', 'zh', 'en', 'The dog runs.'),
            ('안녕하세요.', 'ko', 'en', 'Hello.'),
        ]:
            with self.subTest(language=language, target=target):
                result = self.prepare(self.engine(source, language, target, translated))
                self.assertEqual((result.task_type, result.operation_subtype), ('content_edit', 'replace'))
                self.assertEqual(result.prepared_params['orig'], source)
                self.assertEqual(result.prepared_params['new'], translated)
                self.assertEqual(result.prepared_params['source_language'], language)
                self.assertEqual(result.prepared_params['target_language'], target)
                self.assertEqual(result.prepared_params['instruction_language'], 'en')
                self.assertEqual(result.language, 'en')
                self.assertIn(translated, result.instruction)
                self.assertEqual([x.stage for x in result.llm_calls], ['classify', 'translate_content'])
                self.assertEqual(result.duration_source, 'translation_reference_tts')

    def test_full_translation_uses_official_tts_template(self):
        result = self.prepare(self.engine())
        self.assertEqual(result.instruction, 'Say the following with the same voice: "狗在跑。"')
        self.assertEqual(result.prepared_params['generation_mode'], 'reference_tts')
        self.assertEqual(result.gen_text, '狗在跑。')
        self.assertEqual(result.ref_text, 'The dog runs.')

    def test_full_translation_preserves_complete_reference(self):
        engine = self.engine()
        engine._resolve_audio_duration = Mock(side_effect=AssertionError('must not trim reference'))
        with tempfile.NamedTemporaryFile(suffix='.wav') as f, \
                patch.object(pe, '_audio_duration', return_value=8.0), \
                patch.object(pe, '_prepare_audio', side_effect=AssertionError('must not preprocess reference')):
            result = engine.prepare('Translate into Chinese', f.name)
            self.assertEqual(result.audio, f.name)
        engine._resolve_audio_duration.assert_not_called()

    def test_full_tts_duration_uses_target_not_source_length(self):
        engine = self.engine(translated='你好，欢迎使用语音编辑系统。')
        a = self.prepare(engine).gen_seconds
        b = self.prepare(self.engine(source='A very long source sentence. ' * 100,
                          translated='你好，欢迎使用语音编辑系统。')).gen_seconds
        self.assertEqual(a, b)

    def test_normal_replace_does_not_translate(self):
        engine = self.engine()
        engine._call_llm = Mock(return_value=call('classify', {
            'supported': True, 'task_type': 'content_edit', 'operation_subtype': 'replace',
            'params_extracted': {'orig': 'dog', 'new': 'cat'}, 'language': 'en'}))
        result = self.prepare(engine)
        self.assertIn('cat', result.instruction)
        self.assertEqual(engine._call_llm.call_count, 1)
        self.assertEqual(result.duration_source, 'content_ratio')

    def test_normal_replace_still_requires_orig_new(self):
        engine = self.engine()
        engine._call_llm = Mock(return_value=call('classify', {
            'supported': True, 'task_type': 'content_edit', 'operation_subtype': 'replace',
            'params_extracted': {'orig': 'dog'}, 'language': 'en'}))
        with self.assertRaisesRegex(pe.PromptEnhancerError, 'new'):
            self.prepare(engine)

    def test_second_sentence_only(self):
        result = self.prepare(self.engine(source='First sentence. Second sentence! Third sentence.',
            translated='第二句话！', translation_scope='partial', selector_type='sentence_index', sentence_index=2))
        self.assertEqual(result.prepared_params['orig'], 'Second sentence!')
        self.assertNotIn('First sentence', result.instruction)
        self.assertNotIn('Third sentence', result.instruction)
        start, end = result.prepared_params['source_span']
        self.assertEqual('First sentence. Second sentence! Third sentence.'[start:end], 'Second sentence!')

    def test_unique_source_excerpt(self):
        result = self.prepare(self.engine(translation_scope='partial', selector_type='source_text', source_text='dog'))
        self.assertEqual(result.prepared_params['orig'], 'dog')

    def test_bad_partial_selectors(self):
        for extra in [
            dict(selector_type='source_text', source_text='absent'),
            dict(selector_type='sentence_index', sentence_index=2),
            dict(selector_type='sentence_index', sentence_index=True),
            dict(selector_type='time'), dict(selector_type='speaker'), dict(selector_type='language'),
        ]:
            with self.subTest(extra=extra), self.assertRaises(pe.PromptEnhancerError):
                self.prepare(self.engine(translation_scope='partial', **extra))

    def test_repeated_source_is_ambiguous(self):
        with self.assertRaises(pe.PromptEnhancerError):
            self.prepare(self.engine(source='Hi. Hi.', translation_scope='partial', selector_type='sentence_index', sentence_index=2))

    def test_full_scope_with_partial_selector_rejected(self):
        with self.assertRaises(pe.PromptEnhancerError):
            self.prepare(self.engine(sentence_index=2))

    def test_spanish_rejected_before_translation(self):
        # Hint matters even if an unsupported ASR incorrectly labels Spanish as English.
        for language in ['es', 'en']:
            engine = self.engine(source='Hola.', language=language, source_language_hint='es')
            with self.assertRaises(pe.UnsupportedRequestError):
                self.prepare(engine)
            self.assertEqual(engine._call_llm.call_count, 1)

    def test_unknown_source_and_missing_target_rejected(self):
        for language, target in [(None, 'zh'), ('nospeech', 'zh'), ('en', None), ('en', 'fr')]:
            engine = self.engine(language=language, target=target)
            with self.assertRaises(pe.PromptEnhancerError):
                self.prepare(engine)
            self.assertEqual(engine._call_llm.call_count, 1)

    def test_hint_mismatch_rejected(self):
        with self.assertRaises(pe.PromptEnhancerError):
            self.prepare(self.engine(source_language_hint='ko'))

    def test_failed_or_empty_asr_rejected(self):
        for transcript, error in [('', None), ('.', None), ('guess', 'unreliable')]:
            engine = self.engine()
            engine._transcribe.return_value = pe.ASRCall('test', transcript, 'en', {}, error)
            with self.assertRaises(pe.PromptEnhancerError):
                self.prepare(engine)
            self.assertEqual(engine._call_llm.call_count, 1)

    def test_bad_translations_rejected(self):
        for target, text in [('zh', ''), ('zh', 'English only'), ('en', '中文')]:
            with self.subTest(target=target), self.assertRaises(pe.PromptEnhancerError):
                self.prepare(self.engine(target=target, translated=text))

    def test_duration_override(self):
        result = self.prepare(self.engine(), target_duration=6.0)
        self.assertEqual(result.duration_source, 'user')
        self.assertAlmostEqual(result.gen_seconds, 6.0, delta=.1)

    def test_translation_duration_fallback_and_bounds(self):
        params = dict(orig='hello', new='你好', source_language='en', target_language='zh', translation_scope='partial')
        value, details = pe._translation_duration(params, 5.0, '')
        self.assertEqual(value, 5.0)
        self.assertIn('fallback', details)
        params['new'] = '你好' * 1000
        value, details = pe._translation_duration(params, 5.0, 'hello')
        self.assertEqual(value, 15.0)

    def test_asr_labels_are_preserved_and_not_guessed(self):
        for lang in ['zh', 'en', 'yue', 'ja', 'ko', 'nospeech']:
            self.assertEqual(pe._sensevoice_language(f'<|{lang}|>text', 'text'), lang)
        self.assertIsNone(pe._sensevoice_language('Hola amigo', 'Hola amigo'))
        self.assertEqual(pe._normalize_language('ko'), None)

    def test_fallback_skips_error_even_with_nonempty_text(self):
        bad = Mock(transcribe=Mock(return_value=pe.ASRCall('bad', 'guess', 'en', {}, 'failed')))
        good = Mock(transcribe=Mock(return_value=pe.ASRCall('good', '안녕', 'ko', {})))
        result = pe.FallbackASRProvider([bad, good]).transcribe('unused')
        self.assertEqual(result.model, 'good')
        self.assertEqual(result.language, 'ko')

    def test_mixed_language_tags_rejected(self):
        engine = self.engine()
        engine._transcribe.return_value = pe.ASRCall('SenseVoiceSmall', 'hello 안녕', 'en',
            {'text': '<|en|>hello<|ko|>안녕'})
        with self.assertRaises(pe.UnsupportedRequestError):
            self.prepare(engine)
        self.assertEqual(engine._call_llm.call_count, 1)

    def test_translation_intent_cannot_run_as_vocal_edit(self):
        engine = self.engine()
        engine._call_llm = Mock(return_value=call('classify', {
            'supported': True, 'task_type': 'vocal_edit',
            'params_extracted': {'edit_intent': 'translation', 'target_language': 'en'},
            'language': 'en'}))
        with self.assertRaises(pe.UnsupportedRequestError):
            self.prepare(engine)
        self.assertEqual(engine._call_llm.call_count, 1)

    def test_scope_missing_and_conflicting_selectors(self):
        for extra in [dict(translation_scope=None),
                      dict(translation_scope='partial', selector_type='source_text',
                           source_text='dog', sentence_index=2),
                      dict(translation_scope='partial', selector_type='sentence_index',
                           sentence_index=1, source_text='dog')]:
            with self.subTest(extra=extra), self.assertRaises(pe.PromptEnhancerError):
                self.prepare(self.engine(**extra))

    def test_prompt_contract_for_song_and_dialects(self):
        # Prompt/schema assertions, NOT a live LLM classification accuracy test.
        prompt = pe._build_classify_prompt()
        for term in ['歌曲', '东北口音', '上海话', 'Taiwanese language', 'accent_edit']:
            self.assertIn(term, prompt)
        self.assertNotIn('speech_translation', pe._TASKS)
        self.assertEqual(pe._TASKS['content_edit']['subtypes'],
            ['insert_before', 'insert_after', 'delete', 'delete_before', 'delete_after', 'replace'])

    def test_nontranslation_classifications_do_not_call_translator(self):
        for task, supported in [('accent_edit', True), ('unsupported', False)]:
            engine = self.engine()
            engine._call_llm = Mock(return_value=call('classify', {
                'supported': supported, 'task_type': task, 'params_extracted': {}, 'language': 'zh'}))
            if supported:
                self.assertEqual(self.prepare(engine).task_type, 'accent_edit')
            else:
                with self.assertRaises(pe.UnsupportedRequestError):
                    self.prepare(engine)
            self.assertEqual(engine._call_llm.call_count, 1)


if __name__ == '__main__':
    unittest.main()
