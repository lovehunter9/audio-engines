"""Run the sealed official feature-method bodies without importing a GPU runner."""
import ast
import hashlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
from transformers import WhisperFeatureExtractor

ROOT=Path('/home/olares/s2st-original-check-20261123')


class OfficialFeatureCache:
    def __init__(self,directory,expected_source_sha256):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        source=ROOT/'envs/tencent-model/lib/python3.12/site-packages/simulst_mm/model/processing.py'
        raw=source.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected_source_sha256:
            raise RuntimeError('Official feature source differs from the sealed full baseline')
        module=ast.parse(raw)
        owner=next(node for node in module.body if isinstance(node,ast.ClassDef)
            and any(isinstance(fn,ast.FunctionDef) and fn.name=='_mel_features' for fn in node.body))
        methods=[fn for fn in owner.body if isinstance(fn,ast.FunctionDef) and fn.name in ['_mel_length','_mel_features']]
        shell=ast.ClassDef(name='SealedFeatureMethods',bases=[],keywords=[],body=methods,decorator_list=[])
        scope={'np':np,'torch':torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[shell],type_ignores=[])),str(source),'exec'),scope)
        model=ROOT/'models/Hy-SimulMT-30B-A3B'
        config=json.loads((model/'config.json').read_text())['audio_config']
        if (config.get('fsk_size',2),config.get('hop_length',160),config.get('sample_rate',16000))!=(2,160,16000):
            raise RuntimeError('Unexpected audio token grid for the frozen model')
        extractor=WhisperFeatureExtractor.from_pretrained(model,local_files_only=True)
        from simulinfer.budget import speech_tokens
        self.processor=scope['SealedFeatureMethods']()
        self.processor.info=SimpleNamespace(get_feature_extractor=lambda:extractor,
            get_hop_length=lambda:config.get('hop_length',160),
            get_sampling_rate=lambda:config.get('sample_rate',16000),tokens_per_chunk=speech_tokens)
        self.known={};self.files=set();self.computations=0;self.hits=0
        self.source_sha256=expected_source_sha256

    def ensure(self,path):
        path=Path(path)
        if path.name in self.known:
            self.hits+=1;return
        waveform,rate=sf.read(path,dtype='float32',always_2d=False)
        if rate!=16000 or waveform.ndim!=1:raise RuntimeError('Noncanonical model audio item')
        waveform=np.ascontiguousarray(waveform,dtype='<f4')
        raw=waveform.tobytes()
        fingerprint=14695981039346656037
        for byte in raw:fingerprint=((fingerprint^byte)*1099511628211)&0xffffffffffffffff
        target=self.directory/(f'{fingerprint:016x}-{len(waveform)}.official-mel')
        features,_=self.processor._mel_features([waveform])
        matrix=np.ascontiguousarray(features[0].numpy(),dtype='<f4')
        payload=b'OFTRv001'+struct.pack('<qqq',len(waveform),*matrix.shape)+raw+matrix.tobytes()
        if target.exists():
            if target.read_bytes()!=payload:raise RuntimeError('Feature-cache key collision or content change')
        else:
            temporary=target.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(target)
            self.files.add(target)
        self.known[path.name]=target;self.computations+=1

    def clear(self):
        for path in self.files:
            if path.parent!=self.directory:raise RuntimeError('Invalid owned feature path')
            path.unlink(missing_ok=True)
        self.files.clear();self.known.clear()

    def receipt(self):
        return {'official_processing_source_sha256':self.source_sha256,
            'unchanged_method_bodies':['_mel_length','_mel_features'],
            'feature_computations':self.computations,'cached_audio_hits':self.hits,
            'native_decoder_input_guard':'Full waveform bytes must match before native encoder reads official float32 features',
            'learned_model_or_GPU_inference_in_feature_component':False}
