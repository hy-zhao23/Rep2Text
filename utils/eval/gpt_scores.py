from utils.core.logger import logger
from utils.core.settings import p_args
import openai
import re
import traceback
import os

class GPTScorer:
    DOMAINS = ("wikipedia", "clinical")

    def __init__(self, gt_sen, gen_sen, domain="wikipedia"):
        if domain not in self.DOMAINS:
            raise ValueError(f"domain must be one of {self.DOMAINS}, got '{domain}'")
        self.gt_sen = gt_sen
        self.gen_sen = gen_sen
        self.domain = domain
        self.client = openai.OpenAI(api_key = p_args.settings.openai_key)  
        self.structure_score = None
        self.entity_score = None
        self.topic_score = None

    @property
    def structure_prompt(self):
        
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}

            [GEN]: {self.gen_sen}

            Evaluate the **Structural Frame Similarity** between these sentences.

            Focus on whether [GEN] preserves the core grammatical structure and sentence skeleton of [GT]:
                - Same basic clause structure (e.g., "X was Y of Z" pattern maintained)
                - Same number and type of major phrases (noun phrases, verb phrases in similar positions)
                - Similar syntactic relationships between components
                - Consistent verb tense/aspect (past vs present, perfect vs simple)

            **Scoring Guidelines:**
                - **5**: Identical structure - with almost exactly the same words in the same order
                - **4**: Identical or nearly identical structure - same sentence pattern with only entity substitutions, minor variations like reordering of parallel elements or addition/removal of one modifier
                - **3**: Moderately similar - core structure maintained but with notable changes (e.g., active to passive, clause reordering)
                - **2**: Somewhat similar - recognizable structural elements but significant differences
                - **1**: Minimally similar - only basic sentence type matches (e.g., both declarative)
                - **0**: Completely different structures

            Answer in the following format:
            [ANS] structure: [score]/5
        """

    @property
    def entity_prompt(self):
        if self.domain == "clinical":
            return self._entity_prompt_clinical
        return self._entity_prompt_wikipedia

    @property
    def _entity_prompt_wikipedia(self):
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Entity/Role Consistency and Plausibility** between these sentences.

            Assess whether entities in [GEN] are appropriate replacements for those in [GT], considering:

            **Entity Type Matching Rules:**
                - **Person**: Same role category (athlete↔athlete, scholar↔scholar), similar fame domain, appropriate gender/nationality for context
                - **Location**: Same geographic level (city↔city, country↔country), reasonable proximity or same region
                - **Date/Time**: Numerical proximity (±5 years = high similarity, ±10 years = moderate, ±20 years = low)
                - **Organization**: Same category (university↔university, company↔company, sports team↔sports team)
                - **Event**: Same type (sports↔sports, war↔war, festival↔festival), similar scale/importance
                - **Product/Work**: Same category (game↔game, book↔book, film↔film), similar medium or genre
                - **Concept**: Same domain (math term↔math term, scientific concept↔related concept)

            **Scoring Guidelines:**
                - **5**: Perfect or near-perfect entity matches - all entities are equivalent or highly similar in type and role
                - **4**: Strong matches - entities are contextually appropriate with minor differences (e.g., similar universities, nearby years)
                - **3**: Moderate matches - entities are plausible but with notable differences (e.g., different sports, ±15 years, different genre)
                - **2**: Weak matches - some entities match type but with significant mismatches (e.g., athlete→politician, city→country)
                - **1**: Poor matches - most entities are wrong type but not completely nonsensical
                - **0**: Nonsensical replacements - entities make no sense in context or are completely wrong types

            **Note**: Minor variations within the same category should NOT be penalized heavily. "Asbury College" → "Southern Methodist University" is a strong match (both universities).

            Answer in the following format:
            [ANS] entity: [score]/5
        """

    @property
    def _entity_prompt_clinical(self):
        return f"""
            You are given two clinical text snippets:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Entity/Role Consistency and Plausibility** between these snippets.

            Assess whether entities in [GEN] are appropriate replacements for those in [GT], considering:

            **Clinical Entity Type Matching Rules:**
                - **Patient Demographics**: Age (±5 years = high similarity, ±10 years = moderate), same biological sex, similar DOB precision
                - **Symptoms / Chief Complaint**: Same symptom category (chest pain↔chest tightness, SOB↔dyspnea, nausea↔vomiting); penalise crossing body systems (e.g., chest↔abdominal)
                - **Diagnosis / Condition**: Same condition class (respiratory infection↔respiratory infection, cardiac↔cardiac, neurological↔neurological); similar severity (acute↔acute, chronic↔chronic)
                - **Medications**: Same drug class or mechanism (antibiotic↔antibiotic, SSRI↔SSRI, diuretic↔diuretic); penalise entirely different classes
                - **Vital Signs / Lab Values**: Numerical proximity (BP ±10 mmHg, O2 sat ±3%, glucose ±30 mg/dL); same parameter type
                - **Anatomical Location**: Same body region or system (left lower lobe↔right lower lobe, abdomen↔abdomen); penalise crossing systems
                - **Dates / Admission Timing**: Numerical proximity (±30 days = high, ±1 year = moderate, >5 years = low)
                - **Medical Procedures**: Same procedure category (surgery↔surgery, imaging↔imaging, IV therapy↔IV therapy)

            **Scoring Guidelines:**
                - **5**: All entities match in type and clinical role — equivalent or near-equivalent replacements
                - **4**: Strong matches — same categories with minor differences (e.g., slightly different age, similar drug class)
                - **3**: Moderate matches — entities are plausible but with notable mismatches (e.g., different body system, different drug class)
                - **2**: Weak matches — some entities align but significant clinical mismatches exist (e.g., symptom↔diagnosis, medication↔lab value)
                - **1**: Poor matches — most entity types are wrong, though not nonsensical in a medical context
                - **0**: Nonsensical — entities are medically incoherent or completely unrelated to clinical text

            Answer in the following format:
            [ANS] entity: [score]/5
        """

    @property
    def topic_prompt(self):
        if self.domain == "clinical":
            return self._topic_prompt_clinical
        return self._topic_prompt_wikipedia

    @property
    def _topic_prompt_wikipedia(self):
        return f"""
            You are given two sentences:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Topical Relevance** between these sentences.

            Assess whether [GEN] maintains the same overall subject matter, theme, or knowledge domain as [GT].

            **Topic Categories to Consider:**
                - Biography/People (life events, achievements, roles)
                - Geography/Locations (places, regions, spatial relationships)
                - Organizations/Institutions (companies, universities, government bodies)
                - Historical/Contemporary Events (wars, elections, ceremonies)
                - Arts & Entertainment (films, music, literature, games)
                - Sports & Athletics (competitions, teams, athletes)
                - Science & Technology (concepts, discoveries, technical topics)
                - Nature & Biology (species, phenomena, ecosystems)
                - Products & Brands (commercial items, creative works)

            **Scoring Guidelines:**
                - **5**: Same specific topic - discussing the exact same subject or very closely related subtopic (e.g., both about Danish history, both about a specific person's career)
                - **4**: Same general topic - within the same domain with minor shifts (e.g., both about video games, both about universities, both about the same organization)
                - **3**: Related topics - share a broader category but different focus (e.g., both entertainment but different media, both geography but different aspects)
                - **2**: Loosely related - some thematic connection but substantially different topics (e.g., both about people but completely different contexts)
                - **1**: Minimally related - only the most general connection (e.g., both mention organizations, both have temporal elements)
                - **0**: Unrelated - completely different topics with no meaningful connection

            **Important**: Topic is about the SUBJECT MATTER, not the specific entities. "1782 in Denmark" and "1789 in Denmark" discuss the same topic (Danish historical events) despite different years.

            Answer in the following format:
            [ANS] topic: [score]/5
        """

    @property
    def _topic_prompt_clinical(self):
        return f"""
            You are given two clinical text snippets:
            [GT]: {self.gt_sen}
            [GEN]: {self.gen_sen}

            Evaluate the **Topical Relevance** between these snippets.

            Assess whether [GEN] maintains the same clinical subject matter, disease domain, or care context as [GT].

            **Clinical Topic Categories to Consider:**
                - Cardiology (chest pain, CHF, HTN, arrhythmia, ECG, pacemaker)
                - Pulmonology / Respiratory (COPD, asthma, pneumonia, SOB, O2 saturation, wheezing)
                - Infectious Disease (sepsis, UTI, fever, antibiotics, culture results, cellulitis)
                - Neurology (stroke, migraine, confusion, dementia, seizure, dysarthria)
                - Gastroenterology (abdominal pain, GI bleed, pancreatitis, nausea, vomiting, bowel)
                - Endocrinology / Metabolic (diabetes, glucose control, hypothyroid, electrolytes)
                - Musculoskeletal / Orthopedics (fracture, joint pain, ORIF, mobility)
                - Mental Health / Psychiatry (anxiety, depression, agitation, SSRI, safety watch)
                - Nephrology / Urology (UTI, urine output, voiding, renal function)
                - General / Post-op / Discharge (post-operative care, vital signs monitoring, discharge planning, ambulation)

            **Scoring Guidelines:**
                - **5**: Same specific clinical topic — same disease, condition, or care scenario (e.g., both about COPD exacerbation, both about post-op day 2 care)
                - **4**: Same clinical domain with minor shifts (e.g., both respiratory but COPD vs. asthma; both cardiac but CHF vs. angina)
                - **3**: Related clinical domains — share a broader category but different focus (e.g., both infectious but pneumonia vs. UTI)
                - **2**: Loosely related — some medical connection but substantially different clinical contexts
                - **1**: Minimally related — both are clinical text but discuss completely different physiological systems
                - **0**: Unrelated — no meaningful clinical or topical connection

            **Important**: Topic is about the clinical SUBJECT MATTER, not specific patient details. Two notes about different patients with COPD discuss the same topic.

            Answer in the following format:
            [ANS] topic: [score]/5
        """

    def gpt_rating(self, max_retries=10):
        self.structure_score = self._gpt_rating(self.structure_prompt, max_retries)
        self.entity_score = self._gpt_rating(self.entity_prompt, max_retries)
        self.topic_score = self._gpt_rating(self.topic_prompt, max_retries)

    def _gpt_rating(self, prompt, max_retries):
        _model = os.environ.get("GPT_ID", "gpt-4.1-mini")

        retry_count = 0        
        while retry_count < max_retries:
            try:
                nli_p_res = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=_model,
                )
                result = nli_p_res.choices[0].message.content
                score = self._extract_score(result)
                if score is None:
                    logger.error(f"Failed to extract score from result: {result}")
                    retry_count += 1
                    continue
                return score
            except Exception as e:
                logger.error(f"in gpt_rating: {e}")
                traceback.print_exc()
                retry_count += 1

        logger.error("Failed to get valid scores after 10 retries")
        return None

    def _extract_score(self, result):
        parts = result.split("[ANS]")
        if len(parts) < 2:
            logger.warning(f"No [ANS] tag found in result: {result}")
            return None
        
        result = parts[-1].strip()
        match = re.search(r'(\d)\s*/\s*5', result)
        if not match:
            return None
        else:
            return int(match.group(1))