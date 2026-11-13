"""现有业务摘要，不推断未观察到的业务关系。"""
def analyze(snapshot):
    cutoff=snapshot.metadata['as_of'];h=snapshot.metadata['horizon'];lag=snapshot.metadata['reporting_lag']
    ships={r['id']:r for r in snapshot.rows('shipments')};completed={};seen=set();result=[]
    if h<=0 or lag<0:raise ValueError('观察或晚报上限无效')
    for c in snapshot.rows('completions'):
        if c['shipment'] not in ships or c['shipment'] in seen:raise ValueError('完成身份未知或重复')
        seen.add(c['shipment']);shipped=ships[c['shipment']]['shipped']
        if not shipped<=c['completed']<=c['known']<=c['completed']+lag:raise ValueError('完成时间或晚报上限无效')
        if c['known']<=cutoff:completed[c['shipment']]=c
    for r in snapshot.rows('shipments'):
        if r['shipped']>cutoff:continue
        c=completed.get(r['id']);mature=cutoff>=r['shipped']+h+lag
        if c and c['completed']<r['shipped']:raise ValueError('完成早于发运')
        label=min(c['completed']-r['shipped'],h) if c else (h if mature else None)
        result.append({'shipment':r['id'],'label':label,'mature':mature})
    return result
