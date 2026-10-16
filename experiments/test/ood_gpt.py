#!/usr/bin/env python
# coding: utf-8


import os
os.environ['TRAIN_TAG'] = 'exp05'
os.environ['EINV_CONFIG'] = 'expansion/setup05.yaml'
os.environ.setdefault('PROJ', os.getcwd())
from utils import p_args

text = [
    "John D., age 56, chest pain.",
    "Pt c/o dizziness and nausea; hx of diabetes.",
    "Admitted 02/17/10, fever, sepsis.",
    "Mary T, DOB 1982-11-09. COPD.",
    "Elderly male, HTN, SOB on exertion.",
    "J. Smith, 43, presented with severe headache.",
    "72yo F, CHF, edema improving on Lasix.",
    "Pt found febrile, BP 90/50, started on antibiotics.",
    "Michael A., admitted 03/12/22 for pneumonia, productive cough.",
    "L. Brown, 29, asthma flare, neb.",
    "Hx COPD, increasing dyspnea past two days.",
    "K. Johnson, born in 1977, diagnosed with pancreatitis.",
    "Pt alert, oriented ×3, reports nausea post-op.",
    "Maria G., 44, c/o abdominal pain x3 days.",
    "T. Lewis, 61yo male, hypertension, blood pressure stable.",
    "Admitted 07/09/16 — UTI, positive urine culture.",
    "J. Kim, 38, migraine with photophobia and vomiting.",
    "Pt afebrile, recovering well after appendectomy.",
    "Elderly F, confusion, hx Alzheimer's, placed on safety watch.",
    "Robert C., 53, chest discomfort, ECG normal.",
    "A. Nguyen, 27, mild epigastric tenderness.",
    "68yo M admitted for CHF exacerbation",
    "Pt denies SOB, endorses mild fatigue.",
    "Linda P, DOB 1990-08-14, asthma controlled",
    "Admitted 03/22/21, stroke, right-sided weakness.",
    "Pt resting comfortably, no acute distress.",
    "J. Allen, 49, presented with persistent cough, dx bronchitis",
    "Fever subsided overnight; continue IV fluids.",
    "T. Moore, 34yo F, c/o sore throat and chills",
    "Pt with Type II DM, glucose 256 mg/dL.",
    "O₂ sat dropped to 88%, placed on nasal cannula.",
    "A. Rogers, 71, admitted for pneumonia, stable vitals.",
    "25yo male, ankle fracture s/p ORIF.",
    "Pt appears pale, reports dizziness on standing.",
    "Karen J., 41, migraine recurrent, prescribed sumatriptan.",
    "Admitted 05/12/18 — dehydration secondary to gastroenteritis.",
    "Pt drowsy but arousable, pain 4/10.",
    "B. Adams, 50, presented with GI bleed.",
    "Pt transferred from ICU, on room air.",
    "64yo male, COPD, productive cough, wheezing noted.",
    "Pt tolerated procedure well, minimal bleeding.",
    "L. Torres, 29, diagnosed with anxiety, started sertraline",
    "Hx HTN, on lisinopril, BP controlled.",
    "Pt diaphoretic, c/o chest tightness during exertion.",
    "J. Patel, 32, influenza A positive, supportive care only.",
    "Elderly F, confusion resolved after fluids.",
    "Admitted 08/04/20 — sepsis, source unclear.",
    "Pt ambulated 20 ft without SOB.",
    "M. Davis, 45, UTI, culture pending.",
    "Abdomen soft, non-tender, no guarding.",
    "Hx alcohol use, tremors improving.",
    "Pt discharged home, stable condition.",
    "S. Evans, 59, COPD flare, on 2L oxygen",
    "Pt alert, oriented, pain managed.",
    "Admitted 09/22/19 — stroke, mild dysarthria.",
    "K. Wilson, DOB 1988-06-01, migraine aura",
    "BP trending upward, plan to resume meds.",
    "77yo male, pneumonia, coarse crackles bilaterally.",
    "Pt c/o abdominal distension, denies vomiting.",
    "N. Young, 36, anxiety, insomnia, started on SSRI.",
    "Admitted 03/09/23, right leg erythema.",
    "Afebrile, ambulating without assistance.",
    "Pt resting comfortably, oxygen discontinued.",
    "G. Lee, 46, gout flare, right toe swollen.",
    "80yo M, dementia, agitated overnight.",
    "CXR shows left lower lobe infiltrate.",
    "Pt denies pain, tolerating diet.",
    "D. Harris, 55, CHF, stable on diuretics.",
    "Pt NPO since midnight, awaiting surgery.",
    "62yo F admitted for syncope, workup pending.",
    "Carter, DOB 1991-09-17, diagnosed with asthma",
    "WBC 14.2, likely infection.",
    "Pt c/o dizziness, orthostatics positive.",
    "M. Wright, admitted 07/14/20 for angina.",
    "Nausea improved after ondansetron.",
    "34yo female, fever, chills, probable UTI.",
    "Pt tearful, reports increased anxiety recently.",
    "A. Perez, 42, diagnosed with ulcer, started omeprazole",
    "Admitted 11/05/18, anemia and fever",
    "Pt voiding freely, urine clear.",
    "J. Hall, 65, COPD, oxygen saturation 92% RA",
    "Denies pain, ambulating independently.",
    "28yo male, chest wall pain, musculoskeletal.",
    "Pt with flu symptoms ×3 days.",
    "Hx depression, mood improved.",
    "M. Foster, 49, palpitations, sinus tachy.",
    "BP stable, HR 78, afebrile.",
    "Admitted 10/13/21 — pancreatitis, NPO.",
    "Pt post-op day 2, incision clean, dry, intact.",
    "S. Thomas, 53yo F, on levo for hypothyroid",
    "Diabetic foot ulcer dressed daily.",
    "Pt verbalized understanding of discharge instructions.",
    "J. Brooks, 39, pneumonia, antibiotics ongoing.",
    "Mild wheezing persists despite inhaler use.",
    "Hx asthma, last exacerbation two months ago.",
    "70yo male, bradycardia, pacemaker functioning normally.",
    "Pt's appetite improving, tolerating solid food.",
    "L. Scott, 46, depression, referred to psych.",
    "Admitted 04/06/19 — GI bleed, stable post transfusion",
    "Pt discharged 02/20/22, follow-up in one week."
]

# Cache embeddings
from transformers import AutoTokenizer

emb_id = p_args.model.emb_id or p_args.model.dec_id

tokenizer = AutoTokenizer.from_pretrained(
    emb_id,
    cache_dir=p_args.path.weight_dir,
    token=p_args.settings.hf_token,
    use_fast=True,
    padding_side='right'
)

eos_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

tokens = tokenizer(
    text, 
    add_special_tokens=False, 
    padding=True,
    return_tensors="pt"
)


input_ids = tokens["input_ids"].tolist()
attention_mask = tokens["attention_mask"]
from transformer_lens import HookedTransformer
import torch as th
from utils.data.embeddings import _get_embeddings

model = HookedTransformer.from_pretrained(
    emb_id,
    device="cuda", 
    n_devices=1,
    cache_dir=p_args.path.weight_dir
)

embeddings = _get_embeddings(th.tensor(input_ids), model, attention_mask)


# load model
from utils.models.model import EInvLlamaForCausalLM, EInvConfig
from tqdm import tqdm
from utils import logger
from utils.models.builder import load_projector_weights
from utils.core.path import _get_adapter_dir
from utils.eval import GPTScorer

model_dir = os.path.join(os.getenv("PROJ"), p_args.path.model_dir)
print(model_dir)

device = "cuda" if th.cuda.is_available() else "cpu"

config = EInvConfig.from_pretrained(model_dir)
model = EInvLlamaForCausalLM.from_pretrained(model_dir, config=config, low_cpu_mem_usage=True, torch_dtype=th.bfloat16, local_files_only=True)
model.initialize_einv_tokenizer(tokenizer)


logger.info("Loading pretrained embedding projector weights...")
proj_dir = _get_adapter_dir()
load_projector_weights(model, os.path.join(proj_dir, "projector.bin"))
model = model.to(device)
th.cuda.empty_cache()
from utils.eval.scores import compute_rouge_score, compute_bertscore

# Evaluate

results = []

with th.no_grad():
    for i, emb in enumerate(tqdm(embeddings[:], desc=f"Evaluating")):
        th.cuda.empty_cache()

        id = th.tensor(input_ids[i]).unsqueeze(0)
        emb = th.tensor(emb).unsqueeze(0).to(device)
        attn_mask = id.ne(tokenizer.pad_token_id).to(device)
        id = id.to(device)

        generated = model.generate(
            input_ids=id,
            hidden_embeds=emb,
            attention_mask=attn_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=24,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )


        # Ground-truth continuation: labels already align to next tokens
        greedy_ids = generated[0, 16:].tolist()
        greedy_text = tokenizer.decode(greedy_ids, skip_special_tokens=True)

        # Get ground truth IDs and truncate at EOS token
        eos_token_id = tokenizer.eos_token_id

        # Find EOS token position and truncate
        eos_pos = (id[0] == eos_token_id).nonzero(as_tuple=True)[0]
        if eos_pos.numel() > 0:
            # Truncate at first EOS token
            gt_ids = id[0][:eos_pos[0].item()]

        gt_ids = id[0][:16]
        gt_text = tokenizer.decode(gt_ids.tolist(), skip_special_tokens=True)

        _res = {
            'idx': i,
            "gt_ids": gt_ids.tolist(),
            "greedy_ids": greedy_ids,
            "gt_text": gt_text,
            "greedy_text": greedy_text,
        } 

        rouge_scores = compute_rouge_score([_res])
        bertscore_scores = compute_bertscore([_res])

        _res.update({
            "ROUGE-1": rouge_scores["rouge1"][0],
            "ROUGE-2": rouge_scores["rouge2"][0],
            "ROUGE-L": rouge_scores["rougeL"][0],
            "BERTScore F1": bertscore_scores["f1"][0],
        })

        scorer = GPTScorer(gt_text, greedy_text, domain="clinical")
        scorer.gpt_rating()
        _res.update({
            "structure_score": scorer.structure_score,
            "entity_score": scorer.entity_score,
            "topic_score": scorer.topic_score
        })

        results.append(_res)


import csv

csv_path = os.path.join(os.getenv("PROJ"), "results", "ood_texts.csv")
with open(csv_path, mode="w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Index", "Ground Truth", "Generated"])
    for res in results:
        writer.writerow([res.get("idx", ""), res.get("gt_text", ""), res.get("greedy_text", "")])


from utils.core.files import write_json
json_path = os.path.join(os.getenv("PROJ"), "results", "ood.json")
write_json(results, json_path)
