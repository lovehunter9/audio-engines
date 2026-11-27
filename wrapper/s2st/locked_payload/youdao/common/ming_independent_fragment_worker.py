"""Explicit candidate: constant seed for each independent native generation.

No generated history is provided. A preceding fragment must therefore not
select the following fragment's noise via an incidental shared RNG position.
Keep seed value0 and all quality parameters; restore native ceiling1000.
Prior ceiling-only candidate is separately preserved and rejected.
"""
import json


def main():
    import torch
    import ming_tts
    original=ming_tts.MingTTS.synthesize

    def independent(self,*args,**kwargs):
        torch.manual_seed(0)
        kwargs['max_decode_steps']=1000
        return original(self,*args,**kwargs)

    ming_tts.MingTTS.synthesize=independent
    print(json.dumps({'candidate':'independent-native-fragment',
        'seed_value':0,'seed_scope':'each native synthesize call; no acoustic history',
        'max_decode_steps':1000,'quality_parameters_changed':False,
        'runtime_control_changes':['RNG reset scope','remove artificially short cap'],
        'requires_new_validation':True,'seven_frozen_source_files_modified':False}),flush=True)
    try:
        import frozen_s2st.ming_worker as worker
    except ImportError:
        import wrapper.s2st.ming_worker as worker
    worker.main()


if __name__=='__main__':
    main()
