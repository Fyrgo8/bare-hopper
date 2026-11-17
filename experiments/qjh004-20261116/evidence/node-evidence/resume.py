import concurrent.futures,csv,itertools,json,os,queue,subprocess
from pathlib import Path
r=Path('/home/fengyunhao/bare-hopper-qjh004-20261116')
p=r/'results/coverage-20261116T131616Z'
m=json.loads((p/'manifest.json').read_text())
m['execution']='Resumed with four GPU workers after stopping own serial driver; completed jobs reused'
m['worker_gpus']=[0,3,4,5]
gpus=queue.Queue()
for gpu in m['worker_gpus']:
 assert not subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 gpus.put(gpu)
done={j['label'] for j in m['jobs']}
patterns=m['patterns']; seeds=m['seeds']
tasks=[(s,w,pattern,seed) for s,w in ((1,0),(0,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(4,1)) for pattern,seed in itertools.product(patterns,seeds) if f's{s}-ws{w}-{pattern}-{seed}' not in done]
def work(t):
 s,w,pattern,seed=t;label=f's{s}-ws{w}-{pattern}-{seed}'
 gpu=gpus.get()
 try:
  argv=[str(p/'gemm_bench'),'--compare','--validate-only','--cache','hot','--cases',str(p/'broad.csv'),'--pattern',pattern,'--seed',str(seed),'--validation-repeats','3','--oracle','fp64','--output',str(p/(label+'.csv'))]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),BARE_GEMM_STAGES=str(s),BARE_GEMM_WARP_SPECIALIZED=str(w),BARE_GEMM_STABLE_LONG_K='1')
  with (p/(label+'.log')).open('w') as log:
   try: code=subprocess.run(argv,env=env,stdout=log,stderr=log,timeout=300).returncode
   except subprocess.TimeoutExpired: code=124
  rows=list(csv.DictReader((p/(label+'.csv')).open()))
  expected=list(csv.DictReader((p/'broad.csv').open()))
  complete=len(rows)==14 and [v['case'] for v in rows]==[v['id'] for v in expected] and code in (0,1)
  co=complete and all(v['candidate_ok']=='1' for v in rows)
  ro=complete and all(v['reference_ok']=='1' for v in rows)
  return dict(label=label,argv=argv,gpu=gpu,exit_code=code,rows=len(rows),expected_rows=14,passed=code==0 and co and ro,candidate_passed=co,reference_passed=ro)
 finally: gpus.put(gpu)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(work,t) for t in tasks]
 for future in concurrent.futures.as_completed(futures):
  j=future.result();m['jobs'].append(j)
  (p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
  print(len(m['jobs']),j['label'],'candidate',j['candidate_passed'],'reference',j['reference_passed'],flush=True)
m['candidate_status']='pass' if all(j['candidate_passed'] for j in m['jobs']) else 'fail'
m['reference_status']='pass' if all(j['reference_passed'] for j in m['jobs']) else 'fail'
m['status']='pass' if all(j['passed'] for j in m['jobs']) else 'fail'
m['shape_pattern_seed_checks']=sum(j['rows'] for j in m['jobs'])
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('DONE',m['candidate_status'],m['reference_status'],m['shape_pattern_seed_checks'],flush=True)
