"""Human-readable experiment registry plus append-only machine-readable events."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from features import ROOT

JOURNAL=ROOT/'reports/idea_journal.json'
EVENTS=ROOT/'reports/idea_events.jsonl'

def now():return datetime.now(ZoneInfo('Europe/Moscow')).isoformat()

def log(idea,event,**data):
    EVENTS.parent.mkdir(parents=True,exist_ok=True)
    with EVENTS.open('a') as f:f.write(json.dumps({'time':now(),'idea':idea,'event':event,**data},ensure_ascii=False)+'\n')

def update(idea,**data):
    registry=json.loads(JOURNAL.read_text())
    registry['ideas'][idea].update(data);registry['updated_at']=now()
    JOURNAL.write_text(json.dumps(registry,ensure_ascii=False,indent=2))
    log(idea,'update',**data);render(registry)

def render(registry=None):
    if registry is None:registry=json.loads(JOURNAL.read_text())
    text='# Журнал гипотез RuCode C\n\n'
    text+='Каждое изменение сопоставляется с указанным контролем. Исторические сравнения разных семейств моделей не доказывают отдельный вклад признака. Режим использования public указан в отчёте каждой серии. Исследования с восстановленными public-метками отмечаются отдельно и не заявляются соответствующими правилам настройки.\n\n'
    text+='| ID | Автор / идея | Контроль | Статус | Чистый AUC / A06 | Канал 8 кГц: AUC / A06 | Баллы public | Новые folds: mean / worst AUC |\n|---|---|---|---|---|---|---|---|\n'
    for key,v in registry['ideas'].items():
        m=v.get('metrics',{});c=m.get('clean',{});a=m.get('channel',{})
        fmt=lambda d:f"{d['auc']:.6f} / {d.get('A06',float('nan')):.6f}" if 'auc' in d else '—'
        research=v.get('research_v2',{});research_text=f"{research['n_folds']}: {research['mean_unseen_auc']:.5f} / {research['worst_unseen_auc']:.5f}" if research else '—'
        text+=f"| {key} | {v['author']}: {v['title']} | {v.get('control','—')} | {v.get('status','planned')} | {fmt(c)} | {fmt(a)} | {v.get('public_score','—')} | {research_text} |\n"
    text+='\n## Изменения и ограничения\n\n'
    for key,v in registry['ideas'].items():
        text+=f"### {key}: {v['title']}\n\n{v.get('change','')}\n\n"
        if 'contribution' in v:text+='Изменение относительно контроля: '+json.dumps(v['contribution'],ensure_ascii=False)+'\n\n'
        if 'secondary_contributions' in v:text+='Дополнительные локальные сравнения: '+json.dumps(v['secondary_contributions'],ensure_ascii=False)+'\n\n'
        if 'public_contribution' in v:text+='Публичное сравнение: '+json.dumps(v['public_contribution'],ensure_ascii=False)+'\n\n'
        if 'research_v2' in v:text+='Новая локальная серия: '+json.dumps(v['research_v2'],ensure_ascii=False)+'\n\n'
        if 'notes' in v:text+=v['notes']+'\n\n'
    text+='\nВсе эпохи и результаты по seed: `ideas_results.json`; события: `idea_events.jsonl`. Обновлено: '+registry['updated_at']+'.\n'
    (ROOT/'reports/IDEA_JOURNAL.md').write_text(text)

def initialize():
    if JOURNAL.exists():return
    historical=json.loads((ROOT/'reports/experiment.json').read_text())['models']
    ideas={}
    for key,name,title,author in [
        ('H01',None,'Средняя ошибка интерполяции','Кирилл'),
        ('H02','scalar','Ошибка с RMS-нормировкой','Кирилл / Codex'),
        ('H03','ratio_cat','Ratio и разности + CatBoost','Кирилл / Codex'),
        ('H04','ratio_mlp','Ratio и разности + MLP','Кирилл'),
        ('H05','spectral_cat','Все глобальные признаки + CatBoost','Кирилл / Codex'),
        ('H06','mlp','Все глобальные признаки + MLP','Кирилл / Codex'),
        ('H07','cnn','Карта спектра и ручных признаков + CNN','Кирилл / Codex'),
        ('H08','recon_only','Masked attention и ошибки восстановления','Кирилл'),
        ('H09','recon_cat','Ошибки attention + глобальные признаки','Кирилл / Codex')]:
        m=historical[name] if name else json.loads((ROOT/'reports/original_scalar.json').read_text())
        if 'auc' not in m:m=m.get('metrics',m)
        if 'per_attack' in m:m={**m,**m['per_attack']}
        ideas[key]={'title':title,'author':author,'status':'historical','control':'—',
            'change':'Исходная серия гипотез; результаты сохранены до текущей серии.',
            'metrics':{'clean':{k:v for k,v in m.items() if k in ['auc','A06','robust_auc']}},
            'notes':'Межсемейное сравнение; вклад отдельного изменения не изолирован.'}
    for i,(name,m) in enumerate([(k,v) for k,v in historical.items() if '+' in k],start=10):
        ideas[f'H{i:02d}']={'author':'Кирилл / Codex','title':'Ансамбль '+name,'control':'H07',
            'status':'historical','change':'Равные веса вероятностей; заранее заданный список кандидатов.',
            'metrics':{'clean':m},'notes':'Веса по public не подбирались; комбинация не вошла в финал.'}
    ideas['H16']={'author':'Кирилл','title':'Разность соседних ratio','control':'—','status':'implemented_not_isolated',
        'change':'Входит в набор ratio-признаков; самостоятельная абляция разности пока не проводилась.'}
    ideas['H17']={'author':'Кирилл','title':'Фиксированная карта через обучаемый эмбеддинг','control':'—','status':'partially_implemented',
        'change':'Реализовано фиксированное усреднение; обучаемое сжатие пока не реализовано.'}
    ideas['H18']={'author':'Кирилл','title':'Порог или clipping финального ответа','control':'—','status':'not_tested',
        'change':'Бинарный порог не применялся, поскольку теряет ранжирование для ROC-AUC.'}
    previous=json.loads((ROOT/'reports/cnn_ablation.json').read_text())
    for key,name,title in [('B00','full74','Исходный ансамбль CNN'),('B01','spectrum64','CNN без ручных признаков'),('B02','zero74','CNN с нулевыми ручными каналами')]:
        ideas[key]={'title':title,'author':'Кирилл / Codex','status':'completed_local',
            'control':'B00' if key!='B00' else '—','change':'Согласованная абляция, 2 seed × 4 эпохи.',
            'metrics':{'clean':previous['variants'][name]['ensemble_metrics']}}
    ideas['B00'].update(public_score=86.04,run_id=168015296,
        notes='Public — исходное финальное обучение; локальная абляция заново обучена, MPS может слегка варьировать.')
    ideas['B01']['run_id']=168022126;ideas['B02']['run_id']=168022139
    definitions=[
        ('T00','Контроль временных координат','Codex','—','Исходная карта плюс две нулевые плоскости; одинаковый stem во всех T-вариантах.'),
        ('T01','Относительная временная координата','Кирилл','T00','Плоскость 2u−1, где u — положение от 0 до 1; длительность не передаётся.'),
        ('T02','Только длительность','Кирилл','T00','Постоянная плоскость log(1+исходная длительность)/log(11); позиция не передаётся.'),
        ('T03','Секунды и длительность','Кирилл','T00','Плоскость времени в секундах/5 и отдельная плоскость длительности. Шкала фиксирована, не нормируется отдельно для каждого файла.'),
        ('T04','Добавление относительного PE в карту','Кирилл','T00','Синусоидальное кодирование позиции добавляется к значениям карты с фиксированной амплитудой 0.1; координатные плоскости нулевые.'),
        ('T05','Добавление PE времени в секундах','Кирилл','T04','Та же амплитуда и формула, но фаза зависит от секунд, а не номера сжатого интервала.'),
        ('R00','Контроль дополнительной 1D-ветки','Codex','—','Исходная CNN и отдельная 1D-ветка с нулевым входом; та же архитектура, что R01.'),
        ('R01','Ограниченная кривизна в отдельной ветке','Codex','R00','q=(d_left−d_right)/(|d_left|+|d_right|+0.01*local_RMS+1e-6), масштабы 1/2/4/8, 384 интервала; q, |q| и локальная энергия.'),
        ('A00','Контроль обучения на двух видах','Codex','—','Две одинаковые копии исходной записи в парном батче; средняя BCE. Контроль вычислений и BatchNorm для A01.'),
        ('A01','Аугментация канала связи','Codex','A00','Вторая копия — фиксированное преобразование 16→8→16 кГц; средняя BCE, без consistency.'),
        ('A02','Аугментация и согласованность','Codex','A01','Те же пары плюс 0.1*(p_original−p_augmented)^2. Отдельно измеряет вклад consistency.')]
    for key,title,author,control,change in definitions:
        ideas[key]={'title':title,'author':author,'control':control,'change':change,'status':'planned'}
    registry={'created_at':now(),'updated_at':now(),'protocol':{
        'seeds':[639320752,639320753],'epochs':4,'schedule_epochs':14,
        'main_split':'previous speaker-disjoint A06 holdout','stress_test':'same heldout records after fixed 16→8→16 kHz transform',
        'time_confirmation':'all six T variants also evaluated with held-out A05, seed 639320752',
        'selection':'No public score tuning. Candidate time chosen by mean worst-attack clean AUC over A06 (2 seeds) and A05 (1 seed); ties prefer T03,T05,T01,T04,T02.',
        'limits':'One speaker partition, few seeds; MPS training not bitwise deterministic. Paired contrasts provide evidence, not statistical proof.'},'ideas':ideas}
    JOURNAL.write_text(json.dumps(registry,ensure_ascii=False,indent=2));render(registry);log('ALL','protocol_frozen',protocol=registry['protocol'])

if __name__=='__main__':initialize()
