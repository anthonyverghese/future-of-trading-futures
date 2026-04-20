"""
bot_v5_backtest.py — Zone resets on trade close, not fixed exit threshold.

Zone lifecycle = trade lifecycle:
- Zone enters when price within 1 pt of level → trade opens
- Zone stays active while trade is open
- Zone resets when trade closes (target/stop/timeout) → can re-enter

Scoring integrated into simulation (skipped entries don't activate zone).
Walk-forward validated.

Usage:
    python -u bot_v5_backtest.py
"""

from __future__ import annotations
import datetime, os, sys, time
from dataclasses import dataclass
import numpy as np, pytz

sys.path.insert(0, os.path.dirname(__file__))
from targeted_backtest import DayCache, load_cached_days, load_day, preprocess_day
from bot_risk_backtest import MULTIPLIER, STARTING_BALANCE, FEE_PTS
from score_optimizer import suggest_weight
from walk_forward import _eod_cutoff_ns, INITIAL_TRAIN_DAYS, STEP_DAYS

_ET = pytz.timezone("America/New_York")

# ═══════════════════════════════════════════════════════════════
# PRECOMPUTE FACTOR ARRAYS (once per day)
# ═══════════════════════════════════════════════════════════════

def precompute_arrays(dc):
    fp = dc.full_prices; ft = dc.full_ts_ns
    nf = len(fp); s = dc.post_ib_start_idx
    fp0 = float(dc.post_ib_prices[0])
    tr = np.zeros(nf); l = s
    for i in range(s, nf):
        w = ft[i] - np.int64(180_000_000_000)
        while l < i and ft[l] < w: l += 1
        tr[i] = (i - l) / 3.0
    r30 = np.zeros(nf)
    for i in range(s, nf):
        ws = int(np.searchsorted(ft, ft[i]-np.int64(1_800_000_000_000), side="left"))
        if ws < i: wp = fp[ws:i+1]; r30[i] = float(np.max(wp)-np.min(wp))
    asp = np.zeros(nf); td = np.zeros(nf); l10 = s
    for i in range(s, nf):
        w10 = ft[i] - np.int64(10_000_000_000)
        while l10 < i and ft[l10] < w10: l10 += 1
        if l10 < i:
            el = (ft[i]-ft[l10])/1e9
            asp[i] = abs(float(fp[i])-float(fp[l10]))/max(el,0.1)
            td[i] = (i-l10)/10.0
    dtl = _ET.localize(datetime.datetime.combine(dc.date, datetime.time(12,0)))
    uo = np.int64(dtl.utcoffset().total_seconds()*1e9)
    em = ((ft+uo)//60_000_000_000%1440).astype(np.int32)
    sm = np.zeros(nf)
    for i in range(s, nf): sm[i] = float(fp[i]) - fp0
    return tr, r30, asp, td, em, sm

# ═══════════════════════════════════════════════════════════════
# TRADE RECORD
# ═══════════════════════════════════════════════════════════════

@dataclass
class T:
    date: datetime.date; level: str; direction: str; entry_count: int
    outcome: str; pnl_usd: float
    et_mins: int; tick_rate: float; session_move: float
    range_30m: float; approach_speed: float; tick_density: float

# ═══════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════

def sc(t, w, cw, cl):
    s = 0
    s += w.get({"IBH":"a","IBL":"b","FIB_EXT_HI_1.272":"c","FIB_EXT_LO_1.272":"d","VWAP":"e"}.get(t.level,""),0)
    cm = {("FIB_EXT_HI_1.272","up"):"f",("FIB_EXT_LO_1.272","down"):"g",("IBL","down"):"h",
          ("VWAP","up"):"i",("IBH","up"):"j",("IBL","up"):"k",("FIB_EXT_LO_1.272","up"):"l",
          ("FIB_EXT_HI_1.272","down"):"m",("VWAP","down"):"n"}
    s += w.get(cm.get((t.level,t.direction),""),0)
    if t.et_mins >= 900: s += w.get("tp",0)
    if 1750 <= t.tick_rate < 2000: s += w.get("ts",0)
    elif t.tick_rate < 500: s += w.get("tl",0)
    if t.entry_count == 1: s += w.get("e1",0)
    elif t.entry_count == 2: s += w.get("e2",0)
    elif t.entry_count == 3: s += w.get("e3",0)
    elif t.entry_count == 5: s += w.get("e5",0)
    m = t.session_move
    if 10<m<=20: s+=w.get("mg",0)
    elif -20<m<=-10: s+=w.get("mr",0)
    elif m<=-50: s+=w.get("ms",0)
    elif 0<m<=10: s+=w.get("mz",0)
    elif m>50: s+=w.get("mb",0)
    if cw>=2: s+=w.get("sw",0)
    elif cl>=2: s+=w.get("sl",0)
    if t.range_30m>75: s+=w.get("vh",0)
    if t.approach_speed>3: s+=w.get("av",0)
    elif t.approach_speed>1.5: s+=w.get("af",0)
    if t.tick_density<5: s+=w.get("dl",0)
    return s

def train(trades):
    if not trades: return {}
    n=len(trades); wc=sum(1 for t in trades if t.outcome=="win"); bl=wc/n*100
    sw=suggest_weight
    stk=[]; cw=cl=0
    for t in trades:
        stk.append((cw,cl))
        if t.pnl_usd>=0: cw+=1;cl=0
        else: cw=0;cl+=1
    def wr(fn):
        sub=[i for i in range(n) if fn(trades[i],stk[i][0],stk[i][1])]
        if len(sub)<30: return bl
        return sum(1 for i in sub if trades[i].outcome=="win")/len(sub)*100
    w={}
    for lv,k in [("IBH","a"),("IBL","b"),("FIB_EXT_HI_1.272","c"),("FIB_EXT_LO_1.272","d"),("VWAP","e")]:
        w[k]=sw(wr(lambda t,cw,cl,l=lv:t.level==l),bl)
    for lv,d,k in [("FIB_EXT_HI_1.272","up","f"),("FIB_EXT_LO_1.272","down","g"),("IBL","down","h"),
                    ("VWAP","up","i"),("IBH","up","j"),("IBL","up","k"),("FIB_EXT_LO_1.272","up","l"),
                    ("FIB_EXT_HI_1.272","down","m"),("VWAP","down","n")]:
        w[k]=sw(wr(lambda t,cw,cl,l=lv,dr=d:t.level==l and t.direction==dr),bl)
    w["tp"]=sw(wr(lambda t,cw,cl:t.et_mins>=900),bl)
    w["ts"]=sw(wr(lambda t,cw,cl:1750<=t.tick_rate<2000),bl)
    w["tl"]=sw(wr(lambda t,cw,cl:t.tick_rate<500),bl)
    w["e1"]=sw(wr(lambda t,cw,cl:t.entry_count==1),bl)
    w["e2"]=sw(wr(lambda t,cw,cl:t.entry_count==2),bl)
    w["e3"]=sw(wr(lambda t,cw,cl:t.entry_count==3),bl)
    w["e5"]=sw(wr(lambda t,cw,cl:t.entry_count==5),bl)
    w["mg"]=sw(wr(lambda t,cw,cl:10<t.session_move<=20),bl)
    w["mr"]=sw(wr(lambda t,cw,cl:-20<t.session_move<=-10),bl)
    w["ms"]=sw(wr(lambda t,cw,cl:t.session_move<=-50),bl)
    w["mz"]=sw(wr(lambda t,cw,cl:0<t.session_move<=10),bl)
    w["mb"]=sw(wr(lambda t,cw,cl:t.session_move>50),bl)
    w["sw"]=sw(wr(lambda t,cw,cl:cw>=2),bl)
    w["sl"]=sw(wr(lambda t,cw,cl:cl>=2),bl)
    w["vh"]=sw(wr(lambda t,cw,cl:t.range_30m>75),bl)
    w["af"]=sw(wr(lambda t,cw,cl:1.5<t.approach_speed<=3),bl)
    w["av"]=sw(wr(lambda t,cw,cl:t.approach_speed>3),bl)
    w["dl"]=sw(wr(lambda t,cw,cl:t.tick_density<5),bl)
    return w

# ═══════════════════════════════════════════════════════════════
# TICK-BY-TICK SIMULATION (zone resets on trade close)
# ═══════════════════════════════════════════════════════════════

def simulate(dc, arrs, tgt_fn, stp, mpl, wts, min_sc, cw, cl):
    """Returns (trades, cw, cl). Scoring integrated — skips don't activate zone."""
    tick_rates, range_30m, asp, td, em, smv = arrs
    prices = dc.post_ib_prices; n = len(prices)
    start = dc.post_ib_start_idx
    fp = dc.full_prices; ft = dc.full_ts_ns
    eod = _eod_cutoff_ns(dc.date)

    lvls = [("IBH",dc.ibh),("IBL",dc.ibl),("FIB_EXT_HI_1.272",dc.fib_hi),
            ("FIB_EXT_LO_1.272",dc.fib_lo)]
    has_vwap = hasattr(dc,'post_ib_vwaps') and dc.post_ib_vwaps is not None

    zone = {lv:False for lv,_ in lvls}
    ec = {lv:0 for lv,_ in lvls}
    if has_vwap: zone["VWAP"]=False; ec["VWAP"]=0

    trades=[]; in_t=False; dpnl=0.0; dcons=0; stopped=False
    t_lv=""; t_dir=""; t_lp=0.0; t_tp=0.0; t_sl=0.0; t_to=0; t_ec=0; t_fac={}

    for j in range(n):
        gi = start+j; pj = float(prices[j]); ens = int(ft[gi])
        if ens >= eod:
            if in_t:
                pnl = (pj-t_lp-FEE_PTS if t_dir=="up" else t_lp-pj-FEE_PTS)*MULTIPLIER
                trades.append(T(dc.date,t_lv,t_dir,t_ec,"timeout",pnl,**t_fac))
                zone[t_lv]=False; in_t=False
                if pnl>=0: cw+=1;cl=0;dcons=0
                else: cw=0;cl+=1;dcons+=1
                dpnl+=pnl
            break

        if in_t:
            closed=False; outcome=""; pnl=0.0
            if ens > t_to:
                pnl=(pj-t_lp-FEE_PTS if t_dir=="up" else t_lp-pj-FEE_PTS)*MULTIPLIER
                outcome="timeout"; closed=True
            elif t_dir=="up":
                if pj>=t_tp: pnl=(t_tp-t_lp-FEE_PTS)*MULTIPLIER; outcome="win"; closed=True
                elif pj<=t_sl: pnl=(-(t_lp-t_sl+FEE_PTS))*MULTIPLIER; outcome="loss"; closed=True
            else:
                if pj<=t_tp: pnl=(t_lp-t_tp-FEE_PTS)*MULTIPLIER; outcome="win"; closed=True
                elif pj>=t_sl: pnl=(-(t_sl-t_lp+FEE_PTS))*MULTIPLIER; outcome="loss"; closed=True
            if closed:
                trades.append(T(dc.date,t_lv,t_dir,t_ec,outcome,pnl,**t_fac))
                zone[t_lv]=False; in_t=False; dpnl+=pnl
                if pnl>=0: cw+=1;cl=0;dcons=0
                else: cw=0;cl+=1;dcons+=1
                if dpnl<=-150: stopped=True
                if dcons>=3: stopped=True
            continue

        if stopped: continue

        # Check fixed levels.
        entered=False
        for lv,lp in lvls:
            if zone[lv] or ec[lv]>=mpl: continue
            if abs(pj-lp)>1.0: continue
            d="up" if pj>lp else "down"
            fac={"et_mins":int(em[gi]),"tick_rate":float(tick_rates[gi]),
                 "session_move":float(smv[gi]),"range_30m":float(range_30m[gi]),
                 "approach_speed":float(asp[gi]),"tick_density":float(td[gi])}
            dummy=T(dc.date,lv,d,ec[lv]+1,"",0,**fac)
            if wts and sc(dummy,wts,cw,cl)<min_sc: continue
            zone[lv]=True; ec[lv]+=1
            tgt=tgt_fn(lv)
            tp=(lp+tgt if d=="up" else lp-tgt); sl=(lp-stp if d=="up" else lp+stp)
            tp=round(tp*4)/4; sl=round(sl*4)/4
            in_t=True; t_lv=lv; t_dir=d; t_lp=lp; t_tp=tp; t_sl=sl
            t_to=ens+900_000_000_000; t_ec=ec[lv]; t_fac=fac
            entered=True; break

        if not entered and has_vwap and not zone.get("VWAP",True) and ec.get("VWAP",0)<mpl:
            vp=float(dc.post_ib_vwaps[j])
            if abs(pj-vp)<=1.0:
                d="up" if pj>vp else "down"
                fac={"et_mins":int(em[gi]),"tick_rate":float(tick_rates[gi]),
                     "session_move":float(smv[gi]),"range_30m":float(range_30m[gi]),
                     "approach_speed":float(asp[gi]),"tick_density":float(td[gi])}
                dummy=T(dc.date,"VWAP",d,ec.get("VWAP",0)+1,"",0,**fac)
                if wts and sc(dummy,wts,cw,cl)<min_sc: pass
                else:
                    zone["VWAP"]=True; ec["VWAP"]=ec.get("VWAP",0)+1
                    tgt=tgt_fn("VWAP")
                    tp=(vp+tgt if d=="up" else vp-tgt); sl=(vp-stp if d=="up" else vp+stp)
                    tp=round(tp*4)/4; sl=round(sl*4)/4
                    in_t=True; t_lv="VWAP"; t_dir=d; t_lp=vp; t_tp=tp; t_sl=sl
                    t_to=ens+900_000_000_000; t_ec=ec["VWAP"]; t_fac=fac

    return trades, cw, cl

# ═══════════════════════════════════════════════════════════════
# FMT
# ═══════════════════════════════════════════════════════════════

def fmt(trades, nd, label=""):
    if not trades: return f"  {label:>50s}  no trades"
    w=sum(1 for t in trades if t.outcome=="win")
    l=sum(1 for t in trades if t.outcome=="loss")
    o=len(trades)-w-l; d=w+l; wr=w/d*100 if d else 0
    pnl=sum(t.pnl_usd for t in trades); ppd=pnl/nd
    eq=STARTING_BALANCE; pk=eq; dd=0.0
    for t in trades: eq+=t.pnl_usd; pk=max(pk,eq); dd=max(dd,pk-eq)
    aw=sum(t.pnl_usd for t in trades if t.outcome=="win")/w if w else 0
    al=sum(t.pnl_usd for t in trades if t.outcome=="loss")/l if l else 0
    return(f"  {label:>50s}  {len(trades):>4} ({len(trades)/nd:.1f}/d) "
           f"{w}W/{l}L/{o}O {wr:>5.1f}%  W:{aw:>+6.1f} L:{al:>+6.1f}  "
           f"PnL {pnl:>+8,.0f} ({ppd:>+5.1f}/d)  DD {dd:>5,.0f}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0=time.time()
    print("="*110)
    print("  BOT V5 — Zone resets on trade close + integrated scoring + walk-forward")
    print("="*110)

    days=load_cached_days(); dcs={}
    for date in days:
        try:
            df=load_day(date); dc=preprocess_day(df,date)
            if dc: dcs[date]=dc
        except: pass
    vd=sorted(dcs.keys()); N=len(vd)
    print(f"\n  {N} days loaded in {time.time()-t0:.0f}s")

    print(f"  Precomputing arrays...", flush=True)
    t1=time.time()
    arrs={}
    for i,date in enumerate(vd):
        arrs[date]=precompute_arrays(dcs[date])
        if (i+1)%50==0: print(f"    {i+1}/{N}...",flush=True)
    print(f"  Done in {time.time()-t1:.0f}s")

    # Per-level targets from v4.
    BT={"IBH":14,"IBL":10,"FIB_EXT_HI_1.272":5,"FIB_EXT_LO_1.272":8,"VWAP":6}

    CFGS={
        "T8/S20": (lambda lv:8, 20.0),
        "per-level/S20": (lambda lv:BT.get(lv,8), 20.0),
    }
    MPL=[5,8,12]
    SCORES=[-2,-1,0,1,2,3]

    # First: simulate WITHOUT scoring to get training data for weights.
    print(f"\n  Simulating without scoring (for weight training)...",flush=True)
    t2=time.time()
    no_score_trades={}
    for cfg,(tf,sp) in CFGS.items():
        all_t=[]; cw=cl=0
        for date in vd:
            dt,cw,cl=simulate(dcs[date],arrs[date],tf,sp,12,None,-99,cw,cl)
            all_t.extend(dt)
            no_score_trades.setdefault(cfg,{})[date]=dt
        print(f"    {cfg}: {len(all_t)} trades ({len(all_t)/N:.1f}/d)",flush=True)
    print(f"  Done in {time.time()-t2:.0f}s")

    # Train full-data weights.
    ref=[]
    for date in vd: ref.extend(no_score_trades["T8/S20"].get(date,[]))
    w_full=train(ref)
    print(f"\n  Full-data weights: {dict((k,v) for k,v in w_full.items() if v!=0)}")

    # Now simulate WITH scoring at each threshold.
    print(f"\n  Simulating with scoring...",flush=True)
    t3=time.time()
    results={}  # (cfg, mpl, min_sc) → [trades]
    for cfg,(tf,sp) in CFGS.items():
        for mpl in MPL:
            for ms in SCORES:
                all_t=[]; cw=cl=0
                for date in vd:
                    dt,cw,cl=simulate(dcs[date],arrs[date],tf,sp,mpl,w_full,ms,cw,cl)
                    all_t.extend(dt)
                results[(cfg,mpl,ms)]=all_t
    print(f"  Done in {time.time()-t3:.0f}s")

    # ═══════════════════════════════════════════════════════════
    # STAGE 1: In-sample
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 1: In-sample (all {N} days)")
    print(f"{'='*110}\n")

    is_r=[]
    for (cfg,mpl,ms),trades in results.items():
        pnl=sum(t.pnl_usd for t in trades)
        is_r.append((f"{cfg} max={mpl} score>={ms}",trades,pnl/N))
    is_r.sort(key=lambda x:x[2],reverse=True)
    for l,t,p in is_r[:20]: print(fmt(t,N,l))

    # ═══════════════════════════════════════════════════════════
    # STAGE 2: Walk-forward OOS
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 2: Walk-forward OOS")
    print(f"{'='*110}")

    seen=set(); wf=[]
    for l,_,_ in is_r:
        if l not in seen: seen.add(l); wf.append(l)
        if len(wf)>=15: break

    oos={}; od=0; k=INITIAL_TRAIN_DAYS
    while k<N:
        tr_days=vd[:k]; te_days=vd[k:k+STEP_DAYS]
        if not te_days: break
        od+=len(te_days)
        # Train weights on training data.
        tr_t=[]
        for d in tr_days: tr_t.extend(no_score_trades["T8/S20"].get(d,[]))
        wt=train(tr_t)
        # Simulate test days with trained weights.
        for label in wf:
            parts=label.split(" max="); cfg=parts[0]
            rest=parts[1].split(" score>="); mpl=int(rest[0]); ms=int(rest[1])
            tf,sp=CFGS[cfg]
            test_t=[]; cw=cl=0
            for d in te_days:
                dt,cw,cl=simulate(dcs[d],arrs[d],tf,sp,mpl,wt,ms,cw,cl)
                test_t.extend(dt)
            oos.setdefault(label,[]).extend(test_t)
        k+=STEP_DAYS

    print(f"\n  {od} OOS days\n")
    oos_r=[(l,t,sum(x.pnl_usd for x in t)/od) for l,t in oos.items()]
    oos_r.sort(key=lambda x:x[2],reverse=True)
    for l,t,p in oos_r: print(fmt(t,od,f"OOS: {l}"))

    # ═══════════════════════════════════════════════════════════
    # STAGE 3: Recent 60 days
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*110}")
    print(f"  STAGE 3: Recent 60 days")
    print(f"{'='*110}")
    recent=vd[-60:]; rn=len(recent)
    pre=[d for d in vd if d<recent[0]]
    pre_t=[]
    for d in pre: pre_t.extend(no_score_trades["T8/S20"].get(d,[]))
    wr=train(pre_t)
    print(f"  {recent[0]} → {recent[-1]} ({rn} days)\n")
    for l,_,_ in oos_r[:10]:
        nm=l.replace("OOS: ","")
        parts=nm.split(" max="); cfg=parts[0]
        rest=parts[1].split(" score>="); mpl=int(rest[0]); ms=int(rest[1])
        tf,sp=CFGS[cfg]
        rt=[]; cw=cl=0
        for d in recent:
            dt,cw,cl=simulate(dcs[d],arrs[d],tf,sp,mpl,wr,ms,cw,cl)
            rt.extend(dt)
        print(fmt(rt,rn,f"RECENT: {nm}"))

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s")

if __name__=="__main__": main()
