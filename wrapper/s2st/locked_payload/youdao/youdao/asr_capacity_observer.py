"""Observe actual vLLM request lengths; never edit inputs or enable truncation."""
import json
import os
from pathlib import Path


def install():
    from vllm.entrypoints.llm import LLM
    original=LLM.generate
    limit=int(os.environ['S2ST_ASR_CONTEXT_CAPACITY'])
    path=Path(os.environ['S2ST_ASR_CAPACITY_RECORD'])
    stats={'context_capacity':limit,'requests':0,'max_prompt_tokens':0,'max_output_tokens':0,
           'truncation_enabled':False,'inputs_or_sampling_modified':False}

    def observed(self,*args,**kwargs):
        sampling=kwargs.get('sampling_params')
        values=sampling if isinstance(sampling,list) else [sampling]
        if any(value is not None and getattr(value,'truncate_prompt_tokens',None) is not None for value in values):
            raise RuntimeError('Input truncation is prohibited in the capacity check')
        outputs=original(self,*args,**kwargs)
        for result in outputs:
            count=len(result.prompt_token_ids or [])
            produced=max((len(item.token_ids) for item in result.outputs),default=0)
            if count+produced>limit:raise RuntimeError('Actual ASR request exceeded the declared context capacity')
            stats['requests']+=1
            stats['max_prompt_tokens']=max(stats['max_prompt_tokens'],count)
            stats['max_output_tokens']=max(stats['max_output_tokens'],produced)
        temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(stats,indent=2)+'\n');temporary.replace(path)
        return outputs
    LLM.generate=observed
