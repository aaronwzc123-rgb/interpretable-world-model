"""Validate representative M3 diagnostics against old records and second seeds.
No extrapolated record is labelled a fresh rollout.
"""
import hashlib,json,sys,time
from pathlib import Path
import torch
REPO=Path('C:/Users/super/Desktop/test/Xdreamer')
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(REPO),str(REPO/'scripts')]
import staged_stage_c_eval as sc
from evaluate_m3_corrected_flip import result_record

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
result={'protocol':'For each actor and each of the four initial configurations, compare all historical '
    'records to the new representative outcome/timing, then freshly execute a second actual seed for '
    'the same configuration in baseline and all four negation conditions. No cached rollouts.', 'actors':[]}
for actor in [0,1]:
    hist=json.loads((OUT/f'm3_actor{actor}_unique200_historical.json').read_text())
    diag=json.loads((OUT/f'm3_actor{actor}_configuration_grounding.json').read_text())
    old={r['seed']:r for r in hist['records']}
    agent,cfg=sc.load(REPO/f'models/memory/memory_experiment_3_m3/actor_seed{actor}_400demo.pt','cuda:0',
        distill_meta_path=REPO/'models/memory/memory_experiment_3_m3/frozen_symbolic_head_meta.json')
    entries=[]
    for row in diag['records']:
        match_old=[]
        for oldrow in hist['records']:
            if oldrow['initial_state_signature']!=row['initial_state_signature']:continue
            for condition in ['baseline','cue_early','cue_mid','cue_late','wall_mid']:
                same=oldrow[condition]=={k:row[condition][k] for k in ['outcome','steps']}
                match_old.append(same)
        config=next(c for c in hist['configuration_summary'] if c['signature']==row['initial_state_signature'])
        second=next(s for s in config['actual_seeds'] if s!=row['actual_seed'])
        repeat_checks=[]
        for condition in ['baseline','cue_early','cue_mid','cue_late','wall_mid']:
            kwargs={}
            if condition!='baseline':
                timing=condition.split('_')[1]
                kwargs={'flip_idx':0 if condition.startswith('cue_') else 1,'flip_start':row['flip_steps'][timing]}
            _,out=sc.rollout(agent._wm,agent._task_behavior,second,cfg.time_limit,eval_seed_mode='direct',**kwargs)
            recorded=result_record(out)
            expected=row[condition]
            repeat_checks.append({'condition':condition,'actual_seed':second,
                'full_grounding_trace_and_outcome_identical':recorded==expected,
                'outcome':recorded['outcome'],'steps':recorded['steps']})
        entries.append({'representative_seed':row['actual_seed'],'second_actual_seed':second,
            'signature':row['initial_state_signature'],'historical_seed_count':len(config['actual_seeds']),
            'historical_outcome_timing_comparisons':len(match_old),'historical_all_identical':all(match_old),
            'fresh_second_seed_comparisons':repeat_checks})
        print('validated',actor,row['actual_seed'],second,all(match_old),all(x['full_grounding_trace_and_outcome_identical'] for x in repeat_checks),flush=True)
    result['actors'].append({'actor':actor,'configurations':entries})
    (OUT/'m3_replay_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('DONE',flush=True)
