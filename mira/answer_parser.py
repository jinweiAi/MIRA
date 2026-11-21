"""MMStar deterministic output normalization. No gold, images or scores accepted."""
import re,unicodedata
VERSION='mmstar_normalize_v4_20261105'
def norm(s):
 s=unicodedata.normalize('NFKC',str(s)).casefold()
 s=re.sub(r'<[^>]+>',' ',s)
 s=re.sub(r'[^\w\s]',' ',s)
 return ' '.join(s.split())
def exact(s,options):
 n=norm(s);hits=[o['label'] for o in options if norm(o['text'])==n]
 return hits[0] if len(hits)==1 else None

def candidate(text,options):
 t=str(text).strip().strip('`* \n\t{}').strip();valid={o['label'] for o in options}
 # Standalone legal option letters are labels, even when option texts are permuted letters.
 if re.fullmatch(r'[A-H][.!?,;]?',t):
  label=t[0]
  return (label,'explicit_letter') if label in valid else (None,'illegal_option')
 bracket=re.fullmatch(r'[\(\[]([A-Ha-h])[\)\]][.!]?',t)
 if bracket:return (bracket.group(1).upper(),'explicit_bracket') if bracket.group(1).upper() in valid else (None,'illegal_option')
 # Exact full option texts precede sentence-prefix inference, preserving articles.
 hit=exact(t,options)
 if hit:return hit,'exact_option'
 m=re.match(r'^(?:\(([A-Ha-h])\)|\[([A-Ha-h])\]|([A-H])(?=$|[\s.,:;!)}]))',t)
 if m:
  label=next(x for x in m.groups() if x is not None).upper()
  if label not in valid:return None,'illegal_option'
  rest=t[m.end():].strip(' \t\n.,:;!)}{[]*`')
  alt=exact(rest,options) if rest else None
  if alt and alt!=label:return None,'label_text_conflict'
  if not rest:return label,'explicit_letter'
  # Sentence-initial A can be an indefinite article; require punctuation or exact option text.
  if m.group(3)=='A' and not alt and not re.match(r'^\s*[:.)]',t[m.end():]):return None,'article_ambiguous'
  if alt or m.group(1) or m.group(2) or re.match(r'^\s*[:.)]',t[m.end():]):return label,'letter_with_text'
 return None,'no_candidate'

def parse(text,options):
 t=str(text).replace('**','').replace('`','').strip()
 t=re.sub(r'<\|[^>]+\|>','',t).strip()
 t=t.replace('【answer】','<answer>')
 candidates=re.findall(r'<answer>\s*((?:(?!<answer>).)*?)\s*</answer>',t,re.I|re.S)
 candidates+=re.findall(r'\\boxed\s*\{([^{}]*)\}',t)
 marks=list(re.finditer(r'(?i)\b(?:(?:the\s+)?(?:final|correct)\s+answer|the\s+answer|answer)\s*(?:is\s*:?\s*|[:=]\s*)',t))
 for i,m in enumerate(marks):candidates.append(t[m.end():marks[i+1].start() if i+1<len(marks) else len(t)])
 if candidates:
  parsed=[candidate(c,options) for c in candidates]
  if any(rule=='label_text_conflict' for _,rule in parsed):return None,'label_text_conflict'
  labels={v for v,_ in parsed if v}
  if len(labels)>1:return None,'conflicting_final_answers'
  if len(labels)==1:return next(iter(labels)),'explicit_final'
 direct,rule=candidate(t,options)
 if direct:return direct,rule
 if rule=='label_text_conflict':return None,rule
 lines=[x.strip() for x in t.splitlines() if x.strip()]
 if len(lines)>1:
  label,rule=candidate(lines[-1],options)
  if label:return label,'final_line_'+rule
 # Explicit option citation in concluding prose, with no semantic inference.
 refs=re.findall(r'(?i)(?:corresponds? to|choose|select|option|choice|match (?:would be|is)|value is)\s*(?:option\s*)?([\(\[]?[A-H][\)\]]?)(?=$|[\s.,:;])',t)
 labels={re.sub(r'[^A-H]','',x.upper()) for x in refs}
 if len(labels)==1 and labels.issubset({o['label'] for o in options}):return next(iter(labels)),'explicit_option_reference'
 chinese=re.findall(r'(?:故选|选择|答案[为是])\s*([A-H])',t)
 if len(set(chinese))==1 and chinese[-1] in {o['label'] for o in options}:return chinese[-1],'explicit_chinese_option'
 # Exact first sentence when it is one full option, followed by explanation.
 first=re.split(r'(?<=[.!?])\s+',t,maxsplit=1)[0]
 h=exact(first,options)
 if h:return h,'first_sentence_exact_option'
 # Recover e.g. "the color scheme is A: White and blue" only with exact full option text.
 explicit=[]
 for m in re.finditer(r'(?<![A-Za-z])([A-D])\s*:\s*([^\n]+)',t):
  label=m.group(1);h=exact(m.group(2),options)
  if h and h==label:explicit.append(label)
 if len(set(explicit))==1:return explicit[0],'embedded_letter_exact_text'
 # Short affirmative declarative answer: recover only a full exact option
 # after the final copula; reject uncertainty, negation and alternatives.
 if len(t)<180 and not re.search(r'\b(?:not|never|maybe|might|could|would|or|perhaps|uncertain|unclear|unsure|unknown)\b|[?]',t,re.I):
  copulas=re.findall(r'\b(?:is|are)\s+(.+?)\s*[.!]?\s*$',t,re.I)
  if copulas:
   hit=exact(copulas[-1],options)
   if hit:return hit,'affirmative_copula_exact_option'
 # Remove only an English article, preserving all content words.
 without_article=re.sub(r'^(?:the|a|an)\s+','',norm(t))
 hits=[o['label'] for o in options if re.sub(r'^(?:the|a|an)\s+','',norm(o['text']))==without_article]
 if len(hits)==1:return hits[0],'article_normalized_exact_option'
 # Only exact option after a fixed concluding phrase; do not infer free-form semantics.
 clean=re.sub(r'^(?:Therefore|Thus|Hence|So)[,:]?\s*','',t,flags=re.I)
 label,rule=candidate(clean,options)
 if label:return label,'conclusion_'+rule
 return None,'no_unambiguous_answer'
