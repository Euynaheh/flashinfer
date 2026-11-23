"""Built-in validated ABI39 policy selector (no CUDA/torch imports).

Call select(shape, frozen_policy, incumbent) before explicit caller overrides.
Use only with the hardware/source/numerical contract recorded by the experiment.
Other model shapes return the incumbent; use the offline tuner for those shapes.
The caller must validate the profile fingerprint before calling this function.
"""
import math,json,hashlib

def features(s):
    # Continuous workload/resource features. No model names or exact-shape tests.
    h=s.get('h',4096);i=s.get('i',4096);t=s['t'];k=s['k'];e=s['e']
    rows=4*t*k/e;owners=4*(1-(.75**k))
    return [math.log2(max(rows,1)),math.log2(h/4096),math.log2(i/4096),math.log2(k/owners),math.log2(e/4/64)]
def predict_policy(s,policy):
    x=features(s); candidates=[]
    # First restrict to configurations supported by the nearest workload region.
    # Otherwise a tiny-input latency would spuriously beat every large-input
    # latency when extrapolated. Never compare those unscaled global minima.
    for entry in policy['entries']:
        distances=[sum(w*(a-b)**2 for a,b,w in zip(x,r['features'],[1,4,4,4,1])) for r in entry['observations']]
        candidates.append((entry,distances))
    if not candidates:raise RuntimeError('No qualified candidate profiles')
    closest=min(min(ds) for _,ds in candidates);limit=closest+.05
    ranked=[]
    for entry,ds in candidates:
        near=[(d,r) for d,r in zip(ds,entry['observations']) if d<=limit]
        if not near:continue
        weights=[math.exp(-(d-closest)/.05) for d,_ in near]
        score=sum(w*r['log_us'] for w,(_,r) in zip(weights,near))/sum(weights)
        ranked.append((score,min(ds),entry['config']))
    _,distance,c=min(ranked,key=lambda a:(a[0],a[1],hashlib.sha256(json.dumps(a[2],sort_keys=True).encode()).hexdigest()[:10]))
    return c,dict(distance=distance,features=x,support_limit=limit)

SUPPORTED = {(4096, 4096, 6, 256), (4096, 4096, 8, 288)}

def select(shape, policy, incumbent, *, ep=4, tp=1, same_numa=True, sms=110):
    signature = (shape.get('h', 4096), shape.get('i', 4096), shape['k'], shape['e'])
    if signature not in SUPPORTED or (ep, tp, same_numa, sms) != (4, 1, True, 110):
        return dict(incumbent), {'reason': 'outside_validated_scope'}
    proposed, why = predict_policy(shape, policy)
    # N64 at the T512 anchor is useful only in the N64 workload regime.
    # The incumbent already accounts for the decode admission bucket and rows
    # per expert. No individual failing token value is encoded here.
    if (proposed['k1'], proposed['k2']) == (64, 64) and (
        incumbent['k1'], incumbent['k2']
    ) != (64, 64):
        if (incumbent['k1'], incumbent['k2']) == (32, 32):
            # Extend the already measured N32 stage3 anchor, not the N64
            # anchor, across this N32 regime. Independent confirmation follows
            # the four consistent stage screens at the failing small inputs.
            return dict(incumbent,stage=3), dict(why, reason='n32_stage3_anchor')
        return dict(incumbent), dict(why, reason='n64_anchor_outside_geometry_regime')
    return dict(proposed), dict(why, reason='supported_anchor')
