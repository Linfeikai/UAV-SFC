"""Audit raw heuristic labels against executed mappings; no parameter updates."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--seed',type=int,default=12000)
    parser.add_argument('--random-fraction',type=float,default=0.2)
    args=parser.parse_args();sys.path.insert(0,args.root)
    from test.cpu_model_eval import load_env_config
    from test.evalu import smart_heuristic_policy
    from core.sfc_env import SFCEnv
    from core.uav import UAVNode
    from main import ALGO_REGISTRY
    torch.set_num_threads(1)
    cfg=load_env_config(args.config);cfg['RECORD_DEPLOYMENT']=True
    env=SFCEnv(config=cfg);model=ALGO_REGISTRY['DIFFUSION'].load(args.model,device='cuda')
    observations=[];raw_actions=[];fixed_actions=[];changed=valid=0;started=time.monotonic()
    for episode in range(args.episodes):
        random.seed(args.seed+episode);np.random.seed(args.seed+episode);torch.manual_seed(args.seed+episode)
        obs,_=env.reset(seed=args.seed+episode);done=False
        while not done:
            use_expert=np.random.random()>=args.random_fraction
            action=smart_heuristic_policy(env);velocity=np.asarray([u.velocity for u in env.uavs]);action[:2*env.N]=np.clip(-velocity/(UAVNode.MAX_ACCELERATION*env.dt_fly),-1,1).reshape(-1)
            if not use_expert:action=env.action_space.sample()
            before={k:v.copy() for k,v in obs.items()};tasks=[(u,len(sfc.vnf_chain)) for u,sfc in env.current_cand_tasks[:env.K]]
            obs,reward,terminated,truncated,info=env.step(action);done=terminated or truncated
            if not use_expert or 'detailed_mappings' not in info:continue
            fixed=action.copy();logits=fixed[env.N*2:].reshape(env.K,env.L,env.N);raw=action[env.N*2:].reshape(env.K,env.L,env.N)
            for k,(ue,count) in enumerate(tasks):
                mapping=info['detailed_mappings'][ue]
                for l in range(count):
                    changed+=int(raw[k,l].argmax()!=mapping[l]);valid+=1
                    logits[k,l]=-0.5;logits[k,l,mapping[l]]=0.5
            if len(observations)<256:observations.append(before);raw_actions.append(action);fixed_actions.append(fixed)
    obatch={k:torch.as_tensor(np.stack([o[k] for o in observations]),device='cuda') for k in observations[0]}
    with torch.no_grad():
        raw_loss=float(model.actor.placement_bc_loss(obatch,torch.as_tensor(np.array(raw_actions),device='cuda')))
        fixed_loss=float(model.actor.placement_bc_loss(obatch,torch.as_tensor(np.array(fixed_actions),device='cuda')))
    result={'status':'complete','seed_start':args.seed,'episodes':args.episodes,'config':args.config,'model':args.model,'environment_changed':False,'training_performed':False,'random_fraction':args.random_fraction,'valid_placements':valid,'raw_top_choice_changed_by_environment':changed,'changed_percentage':100*changed/max(1,valid),'bc_batch_size':len(observations),'raw_target_bc_loss':raw_loss,'executed_target_bc_loss':fixed_loss,'elapsed_seconds':time.monotonic()-started}
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
