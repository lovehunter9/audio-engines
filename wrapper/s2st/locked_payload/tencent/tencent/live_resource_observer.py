"""Bounded NVML sampling without retaining a large telemetry stream."""
import asyncio
import json
import time

import pynvml


class ResourceObserver:
    def __init__(self,path):
        pynvml.nvmlInit()
        self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        self.path=path
        self.last=None
        self.data={'scope':'Whole GPU; from warmed MING through MT/ASR startup and outputs',
                   'interval_seconds':0.2,'samples':0,'peak_used_bytes':0,
                   'minimum_free_bytes':None,'maximum_sample_gap_seconds':0,
                   'process_peaks_bytes':{},'sampling_errors':0}

    async def run(self):
        while True:
            now=time.monotonic()
            try:
                memory=pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                data=self.data
                data['total_bytes']=memory.total
                data['peak_used_bytes']=max(data['peak_used_bytes'],memory.used)
                data['minimum_free_bytes']=min(memory.free,data['minimum_free_bytes'] if data['minimum_free_bytes'] is not None else memory.free)
                data['samples']+=1
                if self.last is not None:data['maximum_sample_gap_seconds']=max(data['maximum_sample_gap_seconds'],now-self.last)
                self.last=now
                for process in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
                    key=str(process.pid)
                    data['process_peaks_bytes'][key]=max(data['process_peaks_bytes'].get(key,0),process.usedGpuMemory)
                if data['samples']%25==0:self.save()
            except pynvml.NVMLError:
                self.data['sampling_errors']+=1
            await asyncio.sleep(0.2)

    def save(self):
        temporary=self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data,indent=2)+'\n')
        temporary.replace(self.path)
