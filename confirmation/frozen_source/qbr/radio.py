"""Shared-bandwidth fading channel, request generation, and airtime service."""
from dataclasses import dataclass, asdict
import numpy as np

SERVICES = ('dialogue', 'summary', 'code')
MIX = np.array([.45, .30, .25])
BASE_DT = .05
DT = .1
HALF = .05
EPS = 1e-7

@dataclass(frozen=True)
class Scenario:
    users: int = 16
    edge_slots: int = 4
    cloud_slots: int = 4
    bandwidth_hz: float = 100_000.
    rho: float = .65
    dt: float = .1
    snr_reference_db: float = 10.
    pathloss_exponent: float = 3.5
    fading_correlation_at_100ms: float = .95
    interference_inr_db: float = 10.
    interference_on_rate: float = .3
    interference_off_rate: float = .7
    csi_delay_s: float = 0.
    fading: bool = True
    interference: bool = True
    payload_scale: float = 1.
    beta: float = 12.
    gamma: float = 1000.
    n: int = 1200
    warmup: int = 200
    split: str = 'test'

    def dictionary(self):
        return asdict(self)


def allocate(backlog, observed_capacity, dt, weights=None, rule='sqrt', average=None):
    """Nonnegative airtime fractions on one shared band, with sum<=1.

    Sqrt uses KKT water filling with per-slot buffer caps. Equal and PF/max-
    weight controls use the same cap/reallocation feasibility rules. Caps are
    based on observed CSI; outdated CSI can waste airtime but cannot create it.
    """
    b = np.asarray(backlog, dtype=float)
    c = np.asarray(observed_capacity, dtype=float)
    w = np.ones_like(b) if weights is None else np.asarray(weights)
    cap = b / (c * dt)
    active = b > 1e-8
    f = np.zeros_like(b)
    if not np.any(active):
        return f
    if cap[active].sum() <= 1:
        f[active] = cap[active]
        return f
    if rule in ('maxweight', 'pf'):
        score = b * c if rule == 'maxweight' else c / np.maximum(average, 1.)
        left = 1.
        for u in np.argsort(-np.where(active, score, -np.inf)):
            if not active[u] or left <= 1e-14:
                break
            f[u] = min(left, cap[u]); left -= f[u]
        return f
    mass = np.sqrt(w * b / c) if rule == 'sqrt' else active.astype(float)
    assert rule in ('sqrt', 'equal')
    remaining = active.copy(); left = 1.
    # A capped user is removed permanently, so at most U iterations are needed.
    while np.any(remaining) and left > 1e-14:
        proposal = left * mass / mass[remaining].sum()
        saturated = remaining & (proposal >= cap)
        if not np.any(saturated):
            f[remaining] = proposal[remaining]
            break
        f[saturated] = cap[saturated]
        left -= f[saturated].sum(); remaining[saturated] = False
    return f

class Channel:
    """Policy-independent physical path, generated on a 50 ms base clock.

    Observations reveal the first base sample of a control slot (possibly old).
    Future base samples determine physical service, never the selected action.
    Separate RNGs make the physical path independent of control-slot length.
    """
    def __init__(self, scenario, seed):
        self.sc = scenario; u = scenario.users
        self.rng = np.random.default_rng(seed + 1_000_003)
        self.irng = np.random.default_rng(seed + 2_000_003)
        self.h = (self.rng.normal(size=u) + 1j*self.rng.normal(size=u))/np.sqrt(2)
        self.on = self.irng.random(u) < scenario.interference_on_rate / (
            scenario.interference_on_rate + scenario.interference_off_rate)
        distance = np.linspace(50., 250., u)
        np.random.default_rng(seed + 3_000_003).shuffle(distance)
        self.snr = 10**(scenario.snr_reference_db/10) * (100/distance)**scenario.pathloss_exponent
        self.blocks = []; self.generated = 0

    def extend(self):
        count = 2048; u = self.sc.users
        noise = (self.rng.normal(size=(count,u)) + 1j*self.rng.normal(size=(count,u)))/np.sqrt(2)
        coins = self.irng.random((count,u)); out = np.empty((count,u))
        r = self.sc.fading_correlation_at_100ms**(BASE_DT/.1)
        on_p = 1-np.exp(-self.sc.interference_on_rate*BASE_DT)
        off_p = 1-np.exp(-self.sc.interference_off_rate*BASE_DT)
        inr = 10**(self.sc.interference_inr_db/10)
        for k in range(count):
            self.h = r*self.h + np.sqrt(1-r*r)*noise[k]
            self.on = np.where(self.on, coins[k] >= off_p, coins[k] < on_p)
            gain = np.abs(self.h)**2 if self.sc.fading else np.ones(u)
            interference = self.on * inr if self.sc.interference else np.zeros(u)
            out[k] = self.sc.bandwidth_hz * np.log2(1+self.snr*gain/(1+interference))
        self.blocks.append(out); self.generated += count

    def sample(self, index):
        while self.generated <= index:
            self.extend()
        return self.blocks[index//2048][index%2048]

    def at(self, tick):
        step = round(self.sc.dt/BASE_DT)
        delay = round(self.sc.csi_delay_s/BASE_DT)
        now = tick*step
        observed = self.sample(max(0,now-delay))
        actual = np.mean([self.sample(now+k) for k in range(step)], axis=0)
        return observed, actual


@dataclass
class Requests:
    arrival: np.ndarray
    service: np.ndarray
    task: np.ndarray
    user: np.ndarray
    quality: np.ndarray


def make_requests(cal, sc, seed):
    rng = np.random.default_rng(seed)
    count = sc.n+sc.warmup
    capacity = sc.edge_slots/(MIX@cal.mu[:,0])+sc.cloud_slots/(MIX@cal.mu[:,1])
    arrival = np.cumsum(rng.exponential(1/(sc.rho*capacity),count))
    service = rng.choice(3,count,p=MIX)
    task = np.array([rng.choice(cal.ids[s,sc.split]) for s in service])
    return Requests(arrival,service,task,rng.integers(0,sc.users,count),rng.beta(7,2.5,count))

def shares(backlog, observed, left=1., order=None):
    f=np.zeros(len(observed))
    if order is None:order=np.lexsort((np.arange(len(observed)),-observed))
    for u in order:
        if backlog[u]>EPS and left>1e-14:
            f[u]=min(left,backlog[u]/(observed[u]*DT));left-=f[u]
    assert f.sum()<=1+1e-10
    return f


class Frame:
    def __init__(self, segments):
        self.segments=segments;self.index=0;self.offset=0.
        self.remaining=sum(hi-lo for j,lo,hi in segments)

    def consume(self, amount, rate, start, accepted, completion, terminal, total, last_service):
        """Executor only. Frames carry known offsets; rate is never a policy input."""
        amount=min(amount,self.remaining);used=0.;unique=duplicate=padding=0.
        while amount>1e-10 and self.index<len(self.segments):
            j,lo,hi=self.segments[self.index];lo+=self.offset
            part=min(amount,hi-lo)
            if j<0:padding+=part
            else:
                assert lo<=accepted[j]+EPS,(j,lo,accepted[j])
                gain=max(0.,lo+part-accepted[j]);unique+=gain;duplicate+=part-gain
                accepted[j]=max(accepted[j],lo+part)
                last_service[j]=start+(used+part)/rate
                if terminal[j] and lo+part>=total[j] and not np.isfinite(completion[j]):
                    completion[j]=start+(used+part)/rate
            used+=part;amount-=part;self.offset+=part;self.remaining-=part
            if hi-lo-part<=1e-10:self.index+=1;self.offset=0.
        assert self.remaining>=-EPS
        return used,unique,duplicate,padding
