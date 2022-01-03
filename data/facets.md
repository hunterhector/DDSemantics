```python
%load_ext autoreload
%autoreload 2
print("Started")
```

    Started



```python
from forte.data import MultiPack, DataPack
from forte.processors.base import PackProcessor

from facets.nli.nli_generator import NLIProcessor, TweakData
from facets.nli_reader import MultiNLIReader
from facets.utils import ProgressPrinter
from forte import Pipeline
from forte.data.caster import MultiPackBoxer
from forte.data.selector import NameMatchSelector
from forte.processors.misc import RemoteProcessor
from forte.processors.writers import PackNameMultiPackWriter

from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from forte.common import Resources
from forte.common.configuration import Config
from forte.processors.base import MultiPackProcessor, PackProcessor

```


```python
model_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
# model_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
is_bart = model_name.split("/")[1].startswith("bart")
```


    Downloading:   0%|          | 0.00/50.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.34k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/878k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/446k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/772 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.52G [00:00<?, ?B/s]



```python
def infer(premise: str, hypothesis: str):
    input_pair = tokenizer.encode_plus(
        premise,
        hypothesis,
        return_token_type_ids=True,
        truncation=True
    )

    input_ids = torch.Tensor(
        input_pair['input_ids']).long().unsqueeze(0).to(device)

    token_type_ids = None
    if not is_bart:
        token_type_ids = torch.Tensor(
            input_pair['token_type_ids']
        ).long().unsqueeze(0).to(device)

    attention_mask = torch.Tensor(
        input_pair['attention_mask']).long().unsqueeze(0).to(device)

    if is_bart:
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=None
        )
    else:
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=None
        )
        
    print(torch.softmax(outputs[0], dim=1)[0].cpu().tolist())
```


```python
find_iter = Pipeline().set_reader(
    MultiNLIReader()
).initialize().process_dataset()
```


```python
from onto.facets import NLIPair, Premise, Hypothesis

pack = None
keywords1 = ["sale", "sold", "sale"]
keywords2 = ["export", "exported"]


while True:
    pack = next(find_iter)
    if any([k in pack.text for k in keywords1]) and any([k in pack.text for k in keywords2]):
        print(pack.get_single(Premise).text)
        print(pack.get_single(Hypothesis).text)
        break

```


```python
from forte.data.ontology.top import Annotation
issubclass(Premise, Premise)
issubclass(Premise, Annotation)
```


```python
nli_iter = Pipeline().set_reader(
    MultiNLIReader()
).add(
    # Call spacy on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8008"
    },
).add(
    # Call allennlp on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8009"
    },
).initialize().process_dataset()
```


```python
from ft.onto.base_ontology import Dependency, PredicateLink, PredicateMention, PredicateArgument

def get_arguments(anno):
    pred_args = {}

    for link in anno.get(PredicateLink):
        try:
            pred_args[link.get_parent().tid].append(link.get_child())
        except KeyError:
            pred_args[link.get_parent().tid] = [link.get_child()]
            
    return pred_args

def print_arguments(pred_args):
    for event_id, args in pred_args.items():
        print("# event: " ,pack.get_entry(event_id).text)
        for a in args:
            print(" - ", a.text)
        print("")
```


```python
pack = next(nli_iter)
```


```python
from ft.onto.base_ontology import Dependency, PredicateLink, PredicateMention, PredicateArgument
from onto.facets import NLIPair, Premise, Hypothesis

nli = pack.get_single(NLIPair)
premise = nli.get_parent()
hypo = nli.get_child()
results = nli.entailment

print(premise.text) 

print_arguments(get_arguments(premise))

print(hypo.text)

print_arguments(get_arguments(hypo))

print(results)

infer(premise.text, hypo.text)
```


```python
infer(
    "Thebes held onto power until the 12th Dynasty, when its first king, Amenemhet Iwho reigned between 1980 1951 b.c. established a capital near Memphis.",
    "The capital near Memphis lasted only half a century before its inhabitants abandoned it for the next capital."
)
```

    [0.0014739478938281536, 0.9591830372810364, 0.039342984557151794]



```python
infer(
    "I burst through a set of cabin doors, and fell to the ground.",
    "I burst through the doors and fell down."
)

infer(
    "I burst through a set of cabin doors, and fell to the ground.",
    "I burst through one door and fell down."
)

infer(
    "I burst through one door, and fell to the ground.",
    "I burst through the doors and fell down."
)

infer(
    "I burst through many doors",
    "I burst through one door"
)

infer(
    "I burst through one door and fell to the ground.",
    "I burst through many doors and fell down."
)

infer(
    "I burst through one door",
    "I burst through many doors"
)

infer(
    "I burst through many doors",
    "I burst through one door",
)

infer(
    "I burst through one door",
    "I burst through"
)

infer(
    "I burst through one door",
    "I burst through many door"  # delibrate grammar error
)

infer(
    "I burst through one door",
    "I burst through door" 
)

infer(
    "I burst through one door",
    "I burst through doors" 
)

# May consider include grammar correction
# https://github.com/PrithivirajDamodaran/Gramformer/
```

    [0.9836878776550293, 0.015330430120229721, 0.0009816423989832401]
    [0.0667276531457901, 0.039354611188173294, 0.8939177393913269]
    [0.9542025327682495, 0.040356531739234924, 0.005440966226160526]
    [0.002131389919668436, 0.009634516201913357, 0.9882341623306274]
    [0.0033515170216560364, 0.02520059421658516, 0.9714478850364685]
    [0.0019638068042695522, 0.017047138884663582, 0.9809890389442444]
    [0.002131389919668436, 0.009634516201913357, 0.9882341623306274]
    [0.9090363383293152, 0.08917054533958435, 0.0017931475304067135]
    [0.0020796670578420162, 0.017810672521591187, 0.9801095724105835]
    [0.930550754070282, 0.06708521395921707, 0.002363980980589986]
    [0.8859459757804871, 0.10530628263950348, 0.008747738786041737]



```python
infer(
    "I burst through one door",
    "I burst through one window" 
)
infer(
    "I burst through one door",
    "I burst through one entrace" 
)
infer(
    "I burst through one door",
    "I burst through one doorknob" 
)
infer(
    "I burst through one large door",
    "I burst through one door" 
)
infer(
    "I burst through one door",
    "I burst through one large door" 
)

```

    [0.004438651259988546, 0.028089294210076332, 0.9674720764160156]
    [0.6296278834342957, 0.28136417269706726, 0.08900793641805649]
    [0.22081460058689117, 0.663259744644165, 0.115925632417202]
    [0.7739865183830261, 0.22510668635368347, 0.00090683379676193]
    [0.02266249991953373, 0.9770776033401489, 0.00025990427820943296]



```python
from facets.kbp_reader import EREReader
from facets.nli.analysis import DebugProcessor
from forte import Pipeline

kbp_paths = [
    "/usr0/home/zhengzhl/workspace/corpora/LDC2017E24_DEFT_ERE_Cross_Doc_Event_Coreference_Training_Data_Annotation/data/eng/nw/",
]
pack_iter = Pipeline().set_reader(
    EREReader()
).add(
    # Call spacy on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8008"
    },
).add(
    # Call allennlp on remote.
    RemoteProcessor(),
    config={
        "url": "http://localhost:8009"
    },
).initialize(
).process_dataset(
    kbp_paths
)

```

    WARNING:root:Re-declared a new class named [ConstituentNode], which is probably used in import.



```python
pack = next(pack_iter)
print(pack.text)

from onto.facets import EventMention, EventArgument, Hopper
from ft.onto.base_ontology import EntityMention
arguments = list(pack.get(EventArgument))
ems = list(pack.get(EntityMention))
evms = list(pack.get(EventMention))
hoppers = list(pack.get(Hopper))

print(pack.pack_name)
for idx, h in enumerate(hoppers):
    l = len(h.get_members())
    if l > 1:
        print(f"{idx}: {l}")

```

    <DOC    id="ENG_NW_001278_20131206_F00013SSC">
    <DATE_TIME>2013-12-06T15:19:25</DATE_TIME>
    <HEADLINE>
    Nelson Mandela shining beacon to freedom-loving peoples: Philippines
    </HEADLINE>
    <AUTHOR>MIGO, Zhao Jiemin</AUTHOR>
    <TEXT>
    Nelson Mandela shining beacon to freedom-loving peoples: Philippines
    
    Nelson Mandela shining beacon to freedom-loving peoples: Philippines
    
    MANILA, Dec. 6 (Xinhua) -- The Philippines mourned on Friday the death of former South African President Nelson Mandela, describing him as a "shining beacon of inspiration to all freedom- loving peoples".
    
    Philippine Presidential Communications Operations Office (PCOO) Secretary Herminio Coloma, Jr. said at a press briefing on Friday, "We grieve over the death of Nelson Mandela, a revered world leader, who led his nation and people to freedom by treading the path of peace."
    
    "He endured decades of imprisonment with unwavering fortitude and perseverance, affirming that taking the peaceful, non-violent path to freedom is one that brings about sustained and enduring fulfillment of a people's aspirations for full emancipation," he said.
    
    He added, "In death as in life, he will always be a shining beacon of inspiration to all freedom-loving peoples."
    
    Meanwhile, the Philippine Department of Foreign Affairs (DFA) said in a statement that the Philippines will forever be honored to have hosted Mandela's visit to the country in 1996.
    
    DFA said, the late President Mandela's extraordinary integrity, his tireless commitment and his grand vision that embraced equality for all even under the most trying of circumstances shall never be forgotten.   Enditem
    </TEXT>
    </DOC>
    
    ENG_NW_001278_20131206_F00013SSC
    0: 4



```python
print(
    pack.pack_name,
    len(arguments),
    len(ems),
    len(evms),
    len(hoppers),
)
```

    ENG_NW_001278_20131206_F00013SSC 10 57 10 7



```python
events = hoppers[3].get_members()

args = {}
for arg in arguments:
    args[arg.get_parent().tid] = arg.role, arg.get_child()

for e in events:
    role, arg_ent = args[e.tid]
    print(e.text, e.begin, e.end, role, arg_ent.text)
```

    ordered 2430 2437 entity their
    compelled 1188 1197 entity They
    urged 1071 1076 entity The factory owners



```python
# NYT_ENG_20130522.0183

# "A factory building that collapsed killing more than 1,000 workers "
# "The April 24 disaster killed 1,127 people"
# "Rana and the factory owners be charged with culpable homicide"

infer(
    "the poorly constructed building",
    "A factory building that collapsed"
)

infer(
    "A factory building that collapsed killing more than 1,000 workers",
    "The disaster killed 1,127 people"
)

infer(
    "The disaster killing more than 1,000 workers",
    "The disaster killed 1,127 people"
)

infer(
    "A factory building that collapsed killing more than 1,000 workers",
    "The disaster killed more than 1,000 people"
)


```

    [0.013990906998515129, 0.9551931023597717, 0.030816039070487022]
    [0.003079950576648116, 0.9967959523200989, 0.0001241652062162757]
    [0.0014055155916139483, 0.9957805871963501, 0.002813890343531966]
    [0.9975023865699768, 0.0022071024868637323, 0.0002904499415308237]



```python
# NYT_ENG_20130522.0183
# report

# "government report issued Wednesday concluded"
# "the government report suggested"
# "the government report concluded"
# "Khandaker’s report recommended"
# "Main Uddin Khandaker, a high-ranking official in Bangladesh’s Home Ministry"

infer(
    "the government report suggested",
    "a high-ranking official in Bangladesh’s Home Ministry’s report recommended"
)

infer(
    "the Bangladesh government report suggested",
    "a high-ranking official in Bangladesh’s Home Ministry’s report recommended"
)

infer(
    "the government report suggested",
    "Khandaker’s report recommended"
)

infer(
    "the government report suggested",
    "government report issued Wednesday concluded"
)

infer(
    "government report issued Wednesday concluded",
    "the government report suggested",
)

infer(
    "government report issued Wednesday",
    "the government report",
)

infer(
    "the government report concluded",
    "the government report suggested",
)

infer(
    "the government report",
    "government report",
)

infer(
    "government report issued Wednesday concluded",
    "Khandaker’s report recommended"
)
```

    [0.0010620341636240482, 0.9982618689537048, 0.0006761549739167094]
    [0.0016578171635046601, 0.9979155659675598, 0.0004265854659024626]
    [0.21081799268722534, 0.38938093185424805, 0.3998010754585266]
    [0.412835031747818, 0.29201486706733704, 0.29515013098716736]
    [0.15876099467277527, 0.45221996307373047, 0.38901904225349426]
    [0.30409225821495056, 0.22227801382541656, 0.47362980246543884]
    [0.1332971602678299, 0.1793607622385025, 0.68734210729599]
    [0.11148013174533844, 0.4878629446029663, 0.40065696835517883]
    [0.028560800477862358, 0.814411461353302, 0.1570277214050293]



```python
infer(
    "the death of former South African President Nelson Mandela",
    "the death of Nelson Mandela"
)

infer(
    "the death of former South African President",
    "the death of Nelson Mandela"
)

infer(
    "In death as in life, he ", # coref?
    "the death of Nelson Mandela",
)

infer(
    "the death of Nelson Mandela",
    "In death as in life, he ", # coref?
)
```

    [0.9919731616973877, 0.006763195153325796, 0.001263651647605002]
    [0.012335306033492088, 0.985895574092865, 0.001769156428053975]
    [0.007835613563656807, 0.9852321147918701, 0.006932266987860203]
    [0.12838879227638245, 0.8016220331192017, 0.0699891746044159]



```python
# https://github.com/facebookresearch/GENRE
# Consider using this to get world knowledge

import pickle
from genre.trie import Trie
from genre.fairseq_model import GENRE

# load the prefix tree (trie)
with open("../data/kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))

# load the model
model = GENRE.from_pretrained("models/fairseq_entity_disambiguation_aidayago").eval()

# generate Wikipedia titles
model.sample(
    sentences=["the death of [START_ENT] former South African President [END_ENT] ."],
    prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
)
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-16-a004ff555bae> in <module>
          1 import pickle
    ----> 2 from genre.trie import Trie
          3 from genre.fairseq_model import GENRE
          4 
          5 # load the prefix tree (trie)


    ModuleNotFoundError: No module named 'genre'



```python

infer(
    "Hundreds of depositors rush to banks following an Eurozone bailout deal for Cyprus.",
    "Cyprus bailout deal sends shock and flock to banks.",
)

infer(
    "Hundreds of depositors rush to banks.",
    "Shock and flock went to banks.",
)

infer(
    "Hundreds of depositors rush to banks.",
    "Shock and flock sent to banks.",
)

```

    [0.8262898921966553, 0.17338044941425323, 0.00032965303398668766]
    [0.5529775619506836, 0.4252437353134155, 0.021778680384159088]
    [0.338712215423584, 0.6469780206680298, 0.014309771358966827]



```python

infer(
    "those six hundred injured are in critical condition",
    "over six hundred injured.",
)

infer(
    "those six hundred injured.",
    "over six hundred injured.",
)

infer(
    "more than six hundred injured.",
    "over six hundred injured.",
)

infer(
    "those six hundred injured.",
    "only six hundred injured.",
) # only plays a key role

infer(
    "those six hundred injured.",
    "six hundred injured.",
)

infer(
    "more than six hundred injured are in critical condition",
    "over six hundred injured.",
)

infer(
    "over six hundred injured.",
    "those six hundred injured are in critical condition",
)

infer(
    "six hundred injured.",
    "those six hundred injured are in critical condition",
)

```

    [0.9536704421043396, 0.04087693244218826, 0.005452644545584917]
    [0.822475790977478, 0.16204124689102173, 0.015482996590435505]
    [0.9896883964538574, 0.008850505575537682, 0.0014610919170081615]
    [0.21071797609329224, 0.7655856013298035, 0.02369638718664646]
    [0.9826071858406067, 0.015922198072075844, 0.0014706216752529144]
    [0.9786317348480225, 0.01909623108804226, 0.002271974692121148]
    [0.00042469234904274344, 0.9993520379066467, 0.0002232991246273741]
    [0.00029871342121623456, 0.9995575547218323, 0.0001436806342098862]



```python
infer(
    "the Bangladeshi capital Dhaka collapsed on Wednesday morning, leaving at least 83 people dead and over six hundred injured.",
    "Bangladesh building collapse leaves 83 dead.",
)
```

    [0.9993374943733215, 0.000629924819804728, 3.247023414587602e-05]



```python
infer(
    "Two bombs ripped through the Kuta area of the Indonesian tourist island of Bali on 12 October 2002, leaving 202 people dead",
    "those killed at Paddy's Irish Bar and the nearby Sari Club",
)
infer(
    "Two bombs ripped through the Paddy's Irish Bar, leaving 202 people dead",
    "those killed at Paddy's Irish Bar and the nearby Sari Club",
)
infer(
    "Two bombs ripped through the Paddy's Irish Bar, leaving 202 people dead",
    "those killed at Paddy's Irish Bar",
)
infer(
    "Two bombs ripped through the Kuta area of the Indonesian tourist island of Bali on 12 October 2002, leaving 202 people dead",
    "those killed at Bali",
)

```

    [9.043671525432728e-06, 0.9999685287475586, 2.2401562091545202e-05]
    [7.37878872314468e-05, 0.9999052286148071, 2.095141462632455e-05]
    [0.9904298186302185, 0.009199539199471474, 0.00037066612276248634]
    [0.9995803236961365, 0.0002997621486429125, 0.00011991485371254385]



```python
infer(
    "Olivetti exported $25 million in embargoed, state-of-the-art, flexible manufacturing systems to the Soviet aviation industry.",
    "Olivetti make these sales of the manufacturing systems for $25 million."
)
infer(
    "Olivetti exported $25 million in embargoed, state-of-the-art, flexible manufacturing systems to the Soviet aviation industry.",
    "He make these sales of the manufacturing systems."
)
infer(
    "Olivetti exported $25 million in embargoed, state-of-the-art, flexible manufacturing systems to the Soviet aviation industry.",
    "She make these sales of the manufacturing systems."
)
infer(
    "Olivetti exported $25 million in embargoed, state-of-the-art, flexible manufacturing systems to the Soviet aviation industry.",
    "However, the legality of these sales of the manufacturing systems is still an open question."
)

```

    [0.8278836607933044, 0.1397707760334015, 0.03234565258026123]
    [0.45247882604599, 0.5231260061264038, 0.02439514361321926]
    [0.4065093398094177, 0.5718949437141418, 0.021595656871795654]
    [4.1693070670589805e-05, 0.9999302625656128, 2.8091044441680424e-05]



```python
# Argument number inference

infer(
    "he killed 23 people",
    "13 people were killed",
)
infer(
    "he killed 13 people",
    "13 people were killed",
)
infer(
    "13 people were killed",
    "he killed 13 people",
)
infer(
    "he killed 23 people",
    "23 people were killed",
)
infer(
    "he killed no people",
    "13 people were killed",
)
```

    [0.0014854575274512172, 0.0038892794400453568, 0.9946252703666687]
    [0.9957035183906555, 0.004092487506568432, 0.0002040132094407454]
    [0.07529017329216003, 0.9238279461860657, 0.0008818779024295509]
    [0.9940856695175171, 0.005639192648231983, 0.00027514202520251274]
    [0.00014953994832467288, 0.00047011300921440125, 0.999380350112915]



```python
# Argument type inference
infer(
    "he killed 13 people",
    "he killed 13 dogs"
)
infer(
    "he killed 13 people",
    "13 dogs were killed"
)
infer(
    "he killed 13 people",
    "he killed 13 living entities."
)
infer(
    "he killed 13 living entities.",
    "he killed 13 people",
)
```

    [0.00027956216945312917, 0.004702039062976837, 0.9950183629989624]
    [0.0004624741559382528, 0.004023177549242973, 0.9955143332481384]
    [0.9742183089256287, 0.01471250131726265, 0.01106908917427063]
    [0.27099326252937317, 0.7155007719993591, 0.013505982235074043]



```python

```
