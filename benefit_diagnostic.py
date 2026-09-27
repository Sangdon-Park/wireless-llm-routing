"""Target-integrated prediction error on all measured task pairs (diagnostic)."""
import json
from pathlib import Path
import numpy as np
from qbr.data import Study
from qbr.radio import SERVICES
from qbr.quality import benefit_gram,integrated_loss


def calculate():
    study=Study();result={}
    for s,service in enumerate(SERVICES):
        indices=study.weights[service+'_indices']
        scores=study.weights[service+'_scores']
        weights=study.weights[service+'_weights']
        budget=int(s==0)
        for route in (0,1):
            np.testing.assert_allclose(np.sort(scores[:,route]),study.forecast.decision.quality[s,route,budget],rtol=0,atol=1e-12)
        truth=np.array([[study.rows[int(i)]['budgets']['strict512' if budget else 'full'][route]['quality'] for route in ('edge','cloud')] for i in indices])
        alpha=study.config['alpha'][service]
        calibrated=alpha*weights+(1-alpha)/scores.shape[0]
        moments=calibrated@scores
        gram=benefit_gram(moments,moments)
        cross=benefit_gram(moments,truth)
        diagonal=np.diag(benefit_gram(truth,truth))
        errors={
            'mean':np.maximum(np.diag(gram)-2*np.diag(cross)+diagonal,0.),
            'prompt':integrated_loss(scores,weights,truth),
            'calibrated':integrated_loss(scores,calibrated,truth)}
        result[service]={'tasks':len(indices),'mean_integrated_squared_benefit_error':{m:float(x.mean()) for m,x in errors.items()},
                         'task_ids':[study.tasks[int(i)]['id'] for i in indices],
                         'per_task_error':{m:x.tolist() for m,x in errors.items()}}
    return result

if __name__=='__main__':
    result=calculate();destination=Path('outputs/benefit_diagnostic.json')
    destination.parent.mkdir(exist_ok=True,parents=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({s:r['mean_integrated_squared_benefit_error'] for s,r in result.items()},indent=2))
