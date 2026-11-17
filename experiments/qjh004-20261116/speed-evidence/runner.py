import csv,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
r=Path('/home/fengyunhao/bare-hopper-qjh004-20261116');p=r/'results/speed-20261116'
p.mkdir()
shutil.copy2(r/'build/gemm_bench',p/'gemm_bench')
(p/'formal.csv').write_bytes((r/'configs/gemm.csv').read_bytes())
(p/'long.csv').write_text('id,m,n,k\nlong32768,65,136,32768\nlong65536,65,136,65536\nlarge_long,1024,1024,16384\n')
manifest={'kind':'diagnostic_timing','acceptance_status':'inconclusive_memory_unlocked','gpu':0,'sm_requested':1800,'memory_locked':False,'binary_sha256':hashlib.sha256((p/'gemm_bench').read_bytes()).hexdigest(),'jobs':[]}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2))
assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
(p/'gpu-before.txt').write_text(subprocess.check_output(['nvidia-smi','-i','0','-q'],text=True))
for label,stable,cases in [('old-a','0','long'),('stable-formal','1','formal'),('stable-long','1','long'),('old-b','0','long')]:
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',BARE_GEMM_STAGES='1',BARE_GEMM_WARP_SPECIALIZED='0',BARE_GEMM_STABLE_LONG_K=stable)
 argv=[str(p/'gemm_bench'),'--compare','--cache','rotating','--cases',str(p/(cases+'.csv')),'--rounds','30','--warmup','32','--output',str(p/(label+'.csv'))]
 telemetry=[];external=[]
 with (p/(label+'.log')).open('w') as log:
  proc=subprocess.Popen(argv,env=env,stdout=log,stderr=log)
  while proc.poll() is None:
   t=int(time.time()*1000)
   val=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_event_reasons.active','--format=csv,noheader,nounits'],text=True).strip()
   telemetry.append({'start_ms':t,'end_ms':int(time.time()*1000),'values':val})
   procs=subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).split()
   external.extend(pid for pid in procs if pid!=str(proc.pid))
   time.sleep(.15)
  code=proc.wait()
 (p/(label+'-telemetry.json')).write_text(json.dumps(telemetry,indent=2))
 manifest['jobs'].append({'label':label,'stable':stable,'exit_code':code,'argv':argv,'external_pids':sorted(set(external))})
 (p/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(label,'exit',code,flush=True)
 if code: raise SystemExit(code)
(p/'gpu-after.txt').write_text(subprocess.check_output(['nvidia-smi','-i','0','-q'],text=True))
print('DONE',flush=True)
