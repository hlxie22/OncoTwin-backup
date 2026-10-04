from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class DemoAnswer:
    answer: str
    followups: tuple[str, ...] = ()
    source_fact_types: tuple[str, ...] = ()


def _q(answer: str, *followups: str, source_fact_types: tuple[str, ...] = ()) -> DemoAnswer:
    return DemoAnswer(answer=answer.strip(), followups=tuple(followups), source_fact_types=source_fact_types)


SUGGESTED_QUESTIONS: dict[str, list[str]] = {
    "baseline": [
        "What does OncoTwin know about my cancer?",
        "How has my cancer been behaving so far?",
        "What does my current trajectory mean?",
        "Would one new scan really change the model?",
    ],
    "after_progression": [
        "What changed on my newest scan?",
        "Why did my outlook change?",
        "What does OncoTwin think is most important right now?",
        "What should I ask my oncologist?",
    ],
    "after_esr1": [
        "What does this new blood test mean?",
        "Does the ESR1 result change my OncoTwin forecast?",
        "What changed after adding this blood test?",
        "What should I ask my oncologist about ESR1?",
    ],
}


COMMON: dict[str, DemoAnswer] = {
    "What does OncoTwin know about my cancer?": _q(
        """
Based on the records you have added, OncoTwin has built a longitudinal view of your cancer rather than treating each report separately. Your verified record describes estrogen-receptor-positive, HER2-negative metastatic breast cancer involving the liver and bone. It also tracks treatment with letrozole and ribociclib, treatment-related neutropenia and dose adjustment, your scan history, and molecular findings that have been added later.

Important facts remain linked to their source records so you can check where they came from. The quantitative trajectory is generated separately by OncoTwin's frozen forecasting model from the supported parts of that verified history.
""",
        "Can you explain my cancer in plain English?",
        "Where has my cancer spread?",
        "How does OncoTwin build my history?",
        source_fact_types=("diagnosis", "er_status", "her2_status", "disease_site", "treatment"),
    ),
    "Can you explain my cancer in plain English?": _q(
        """
Your records describe breast cancer that has spread beyond the breast to other parts of the body, including the liver and bone. The cancer is estrogen-receptor positive, meaning estrogen signaling is one of the biological pathways involved in its growth. It is HER2 negative, meaning it does not have the type of HER2 overexpression that defines HER2-positive breast cancer.

Those features help describe the disease and provide context for the treatment decisions your oncology team makes. OncoTwin uses the verified record to explain the history, but it does not replace your oncology team's diagnosis or treatment decisions.
""",
        "Where has my cancer spread?",
        "What treatment am I on?",
        source_fact_types=("diagnosis", "er_status", "her2_status", "disease_site"),
    ),
    "Where has my cancer spread?": _q(
        """
The verified records currently document metastatic disease in the liver and bone. OncoTwin does not mark an organ as involved merely because it appeared on a scan. A disease site enters the patient state only when the underlying record supports metastatic involvement there.
""",
        "How does OncoTwin build my history?",
        source_fact_types=("disease_site",),
    ),
    "What treatment am I on?": _q(
        """
Your record shows treatment with letrozole and ribociclib. Later records also document treatment-related neutropenia, followed by a hold and dose adjustment of ribociclib.

OncoTwin records those events as part of your cancer history, but it does not decide whether a treatment should be started, stopped, or changed.
""",
        "Can OncoTwin tell me what treatment I should take?",
        "Can you help me prepare for my next appointment?",
        source_fact_types=("treatment", "adverse_event", "laboratory"),
    ),
    "How does OncoTwin build my history?": _q(
        """
OncoTwin reads the records you add, extracts clinically important facts, keeps those facts linked to their source pages, and combines verified information into a longitudinal patient state. New records update that state instead of replacing the earlier history.

The trajectory model then uses the supported model inputs derived from that evolving verified history. The language assistant explains the result; it does not generate the quantitative forecast itself.
""",
        "Where is this information coming from?",
        "Why does the model use all my previous scans?",
    ),
    "Where is this information coming from?": _q(
        """
Patient-specific information shown by OncoTwin comes from the medical records you added. Important facts retain links to the report and page that supported them. The quantitative trajectory comes from a separate frozen forecasting model using the verified patient state.

When OncoTwin cannot establish something from the available record, it should treat that information as unavailable rather than silently assuming a negative result.
""",
        "How certain is OncoTwin?",
        "What information is the model missing?",
    ),
    "Can OncoTwin tell me what treatment I should take?": _q(
        """
No. OncoTwin can help you understand your records, how your disease has changed over time, how the model's estimate changed, and what questions may be useful to discuss with your oncology team. It does not choose treatment for you.
""",
        "What should I ask my oncologist?",
        "Can you help me prepare for my next appointment?",
    ),
    "How certain is OncoTwin?": _q(
        """
There are two different kinds of uncertainty here. First, the medical record itself can be incomplete or ambiguous; OncoTwin keeps uncertain information separate rather than silently guessing. Second, the trajectory is a model estimate, not a prediction of exactly what will happen to you.

The model gives a probability based on the supported representation of your verified history. Individual outcomes can differ, and external confirmation of the deployed model in a new independent cohort is still pending.
""",
        "Could the forecast be wrong?",
        "What information is the model missing?",
    ),
    "Could the forecast be wrong?": _q(
        """
Yes. The trajectory is a research-model estimate, not a clinical certainty. The model may not capture every factor relevant to your disease, and external confirmation in a new independent cohort is still pending.

The estimate is best treated as a structured perspective on the trajectory, not a replacement for your oncology team's judgment.
""",
        "How certain is OncoTwin?",
    ),
    "What information is the model missing?": _q(
        """
OncoTwin can only use information represented faithfully in its supported model inputs. Some clinical information may be absent from the records you have uploaded, and the deployed forecasting model cannot use every type of modern oncology data.

Missing information is treated as unavailable rather than assumed to be negative.
""",
        "How certain is OncoTwin?",
        "Where is this information coming from?",
    ),
    "Why does the model use all my previous scans?": _q(
        """
A single scan tells us what was seen at one moment. A sequence of scans tells us how the cancer has been behaving over time. OncoTwin's model was designed around longitudinal updates, so earlier imaging provides context for the newest supported scan evidence.
""",
        "How certain is OncoTwin?",
    ),
    "What should I ask my oncologist?": _q(
        """
Useful questions depend on where you are in the journey. In general, you can ask your oncologist what the newest verified findings mean, what remains uncertain, whether additional testing would change the picture, and what options or studies are worth discussing.

When a major new scan or molecular result is added, OncoTwin can make those questions more specific to that change.
""",
        "Can you help me prepare for my next appointment?",
    ),
    "Is there research relevant to what's happening to me?": _q(
        """
Yes. OncoTwin can use features of your verified cancer state to retrieve relevant published research and clinical studies. Those sources provide context for discussion with your care team; they do not establish that a treatment or study is appropriate for you.

The Prepare section can run the source-linked evidence search rather than relying on the assistant to invent current literature from memory.
""",
        "Are there clinical trials that might be relevant?",
        "Can you help me prepare for my next appointment?",
    ),
    "Are there clinical trials that might be relevant?": _q(
        """
OncoTwin can surface trials whose published study information appears relevant to verified features in your record, such as ER-positive/HER2-negative metastatic breast cancer, progression, or an ESR1 alteration. It does not determine that you are eligible.

Formal eligibility depends on the complete inclusion and exclusion criteria and should be reviewed by the study team and your oncology team.
""",
        "Am I eligible for one of these trials?",
        "What should I ask my oncologist?",
    ),
    "Am I eligible for one of these trials?": _q(
        """
OncoTwin cannot determine trial eligibility from your record alone. It can show why a study may be worth discussing and which features of your record appear relevant, but formal eligibility requires review of the complete criteria by qualified clinicians or the study team.
""",
        "Are there clinical trials that might be relevant?",
    ),
    "Can you help me prepare for my next appointment?": _q(
        """
Yes. OncoTwin can summarize what has changed in your verified record, highlight the newest scan and molecular findings, and prepare focused questions for your oncology team. The Prepare section can also generate a concise visit brief and source-linked research or trial leads.
""",
        "What should I ask my oncologist?",
        "Is there research relevant to what's happening to me?",
    ),
}


BASELINE: dict[str, DemoAnswer] = {
    "How has my cancer been behaving so far?": _q(
        """
Your record so far shows metastatic breast cancer involving the liver and bone followed over multiple scans while you have been receiving letrozole and ribociclib. Before the newest progression report is added, the recent imaging sequence has generally provided evidence of disease control rather than clear new progression.

OncoTwin uses that sequence, not just a single scan, when constructing the trajectory.
""",
        "What are the most important things in my history right now?",
        "Why does the model use all my previous scans?",
        "Would one new scan really change the model?",
        source_fact_types=("scan_assessment", "treatment", "disease_site"),
    ),
    "What are the most important things in my history right now?": _q(
        """
Three things stand out in the verified history. First, your cancer is ER-positive and HER2-negative with documented liver and bone metastases. Second, you have been receiving letrozole plus ribociclib, with a treatment hold and dose adjustment after neutropenia. Third, the sequence of imaging reports gives the model longitudinal information about how the disease has behaved on treatment.
""",
        "What does my current trajectory mean?",
        "What information is the model missing?",
        source_fact_types=("er_status", "her2_status", "disease_site", "treatment", "scan_assessment"),
    ),
    "What does my current trajectory mean?": _q(
        """
Your trajectory is an estimate of the probability of remaining progression-free over time, based on the cancer history OncoTwin can currently represent. It is not a countdown and it does not tell us exactly when your cancer will progress.

It is better thought of as the model's current view of your disease course given the supported information available at this point in your history.
""",
        "What is driving my current outlook?",
        "How certain is OncoTwin?",
        "Would one new scan really change the model?",
    ),
    "What is driving my current outlook?": _q(
        """
The forecast reflects the supported model representation of your verified longitudinal history, including the temporal imaging pattern available before the next scan. OncoTwin should not interpret that as proof that one individual clinical fact or lesion caused the forecast value.
""",
        "How certain is OncoTwin?",
        "What information is the model missing?",
    ),
    "Would one new scan really change the model?": _q(
        """
It can. The forecasting model is specifically designed to update when new scan evidence becomes available. If a new scan is consistent with the previous pattern, the trajectory may change only modestly. If it contains meaningfully different evidence, such as documented progression, the updated trajectory can shift more substantially.
""",
        "What should I pay attention to on my next scan?",
        "Why does the model use all my previous scans?",
    ),
    "What should I pay attention to on my next scan?": _q(
        """
The important question for OncoTwin is whether the new scan continues the pattern seen in your earlier imaging or provides evidence that the disease is changing. Once a new radiology report is verified, the supported scan evidence is added to the longitudinal state and the trajectory can be updated.
""",
        "Would one new scan really change the model?",
        source_fact_types=("scan_assessment",),
    ),
}


PROGRESSION: dict[str, DemoAnswer] = {
    "What changed on my newest scan?": _q(
        """
The newest CT changes the picture in the liver. The report describes growth of the dominant liver lesion and the appearance of a new liver metastasis. At the same time, the known bone metastases were described as stable.

That combination is why the verified scan state now reflects progression rather than implying that every known site worsened in the same way.
""",
        "What stayed the same?",
        "Why does liver progression matter if my bones are stable?",
        "Why did my outlook change?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "What stayed the same?": _q(
        """
The important stable finding is the known bone metastatic disease. The newest CT describes liver progression, but the bone lesions were not reported as progressing. OncoTwin keeps those observations distinct rather than summarizing the entire scan as uniformly worse.
""",
        "How can part of my cancer be stable while another part progresses?",
        "Why does liver progression matter if my bones are stable?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "Why did my outlook change?": _q(
        """
Before this scan was added, OncoTwin's 6-month progression-free estimate was {PRE_PFS_6M}. After the new scan was incorporated into the same longitudinal history, the updated estimate was {POST_PFS_6M}, a change of {DELTA_6M_PP} percentage points.

The verified new evidence is liver progression on the newest CT. The model is reacting to the updated longitudinal pattern; OncoTwin should not claim that one individual lesion alone caused the forecast change.
""",
        "What part of the scan mattered most to the model?",
        "Could the forecast be wrong?",
        "What changed in my record versus what changed in the model?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "How much did the scan change my outlook?": _q(
        """
At 6 months, the model-estimated progression-free outlook changed from {PRE_PFS_6M} before the scan to {POST_PFS_6M} after it was added, a difference of {DELTA_6M_PP} percentage points.

The full trajectory view shows the before-and-updated curves because the model update is not limited to a single time point.
""",
        "Does the lower forecast mean I will progress within six months?",
        "Could the forecast be wrong?",
    ),
    "What part of the scan mattered most to the model?": _q(
        """
We can say that the model changed after receiving the new structured scan evidence showing progression. We should not claim that the model assigned a causal amount of importance to one particular lesion.

Clinically, the major new findings in the report were growth of the dominant liver lesion and a new liver metastasis, while the known bone disease remained stable.
""",
        "Can you tell which lesion caused the forecast to drop?",
        "Does a bigger change in the forecast mean the scan was more severe?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "Why does liver progression matter if my bones are stable?": _q(
        """
Cancer can behave differently in different disease sites at the same time. Stable bone lesions do not cancel out new evidence of progression elsewhere. In your newest scan, the liver findings indicate a meaningful change even though the known bone disease remained stable.

OncoTwin preserves both pieces of information rather than forcing them into an all-or-nothing description.
""",
        "How can part of my cancer be stable while another part progresses?",
        "Is a new lesion different from an existing lesion getting bigger?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "Does this mean my treatment stopped working?": _q(
        """
The scan documents progression while your current treatment is part of the recorded clinical history, which is important information for your oncology team. OncoTwin does not determine on its own whether a treatment has "stopped working" or whether it should be changed.

That decision depends on the full clinical situation and should be made with your oncologist.
""",
        "What should I ask my oncologist?",
        "What information would be useful next?",
        source_fact_types=("scan_assessment", "treatment"),
    ),
    "Is this definitely bad news?": _q(
        """
The scan contains important unfavorable evidence because liver progression was documented. At the same time, it does not say that every known site worsened; the bone disease remained stable.

The most faithful interpretation is that the disease picture has changed in a clinically meaningful way, rather than reducing the entire report to simply "good" or "bad."
""",
        "What stayed the same?",
        "What does OncoTwin think is most important right now?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "Does the lower forecast mean I will progress within six months?": _q(
        """
No. An updated 6-month estimate of {POST_PFS_6M} is a probability estimate, not a statement that you personally will or will not progress within six months. It describes the model's estimated probability of remaining progression-free given the supported representation of your history.

Individual outcomes can differ substantially from a probability estimate.
""",
        "How certain is OncoTwin?",
        "Could the forecast be wrong?",
    ),
    "Why should I trust the change in the forecast?": _q(
        """
The useful property of the comparison is consistency: the same frozen forecasting model is applied to the longitudinal history before and after the new scan, with the newly verified scan evidence added for the updated estimate.

That does not make the model certain or clinically definitive. It gives you a structured way to see how new evidence changes the model's assessment without letting the language assistant rewrite the prediction.
""",
        "Could the forecast be wrong?",
        "What changed in my record versus what changed in the model?",
    ),
    "Could the forecast be wrong?": _q(
        """
Yes. This is a research-model estimate, not a clinical certainty. The model may not capture every factor relevant to your disease, and external confirmation in a new independent cohort is still pending.

The estimate is best treated as a structured perspective on the trajectory, not a replacement for your oncology team's judgment.
""",
        "How certain is OncoTwin?",
        "What information would be useful next?",
    ),
    "What does OncoTwin think is most important right now?": _q(
        """
The biggest change in the verified record is the transition from the earlier imaging pattern to documented liver progression on the newest CT. That new scan state is also accompanied by a meaningful change in the quantitative trajectory.

Useful topics for the next oncology discussion include how the team interprets the progression, whether additional molecular information would be helpful, and what options or studies are worth discussing.
""",
        "What should I ask my oncologist?",
        "What information would be useful next?",
        "Is there research relevant to what's happening to me?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "What should I ask my oncologist?": _q(
        """
Based on what changed in your record, useful questions include:

1. What does the liver progression on this scan mean for the current treatment plan?
2. Would another biopsy or a blood-based genomic test be useful now that the cancer has progressed?
3. What treatment options are most relevant if the current regimen is no longer adequately controlling the liver disease?
4. Are there clinical trials I should consider at this point?

These are discussion questions, not treatment recommendations from OncoTwin.
""",
        "What information would be useful next?",
        "Are there clinical trials that might be relevant?",
        "Can you help me prepare for my next appointment?",
        source_fact_types=("scan_assessment", "treatment"),
    ),
    "What information would be useful next?": _q(
        """
After documented progression, additional molecular information can sometimes help the oncology team characterize the disease and discuss next steps. OncoTwin can add new verified findings to your longitudinal cancer history.

Not every new clinical finding is necessarily a supported input to the frozen quantitative forecasting model, so the patient-state update and the model update are kept separate.
""",
        "What should I ask my oncologist?",
        "Are there clinical trials that might be relevant?",
    ),
    "What changed in my record versus what changed in the model?": _q(
        """
They are related but different steps. Your record changed because the new CT added verified structured evidence of liver progression. The model changed because that updated supported patient state was then passed through the frozen forecasting system, producing a different trajectory.

The language assistant explains those two outputs. It does not generate or alter the quantitative forecast.
""",
        "Why should I trust the change in the forecast?",
        "Could the forecast be wrong?",
        source_fact_types=("scan_assessment",),
    ),
    "How can part of my cancer be stable while another part progresses?": _q(
        """
Different metastatic sites can behave differently under the same treatment. A scan can therefore contain both stable and progressive findings. Your newest report is an example: the known bone metastases remained stable while the liver showed progression.

OncoTwin keeps disease sites and scan findings distinct so that this mixed picture is not lost.
""",
        "Why does liver progression matter if my bones are stable?",
        source_fact_types=("scan_assessment", "disease_site"),
    ),
    "Is a new lesion different from an existing lesion getting bigger?": _q(
        """
They are different observations. Growth of an existing lesion indicates change in disease that was already visible, while a new metastatic lesion indicates newly visible metastatic disease within that organ. Your report contains both types of liver evidence, which together support the progression assessment in the report.
""",
        "What part of the scan mattered most to the model?",
        source_fact_types=("scan_assessment",),
    ),
    "Did the neutropenia or ribociclib dose reduction cause the progression?": _q(
        """
Your record contains the earlier neutropenia and ribociclib dose adjustment as well as the later progression scan, but that sequence does not establish that one caused the other.

OncoTwin can connect events chronologically. It should not infer a patient-specific causal relationship that the records do not establish.
""",
        "Does this mean my treatment stopped working?",
        "What changed in my record versus what changed in the model?",
        source_fact_types=("treatment", "adverse_event", "laboratory", "scan_assessment"),
    ),
    "Was the model already expecting this progression?": _q(
        """
The earlier trajectory represented probabilities, not a prediction that one specific future scan would show progression. Before the new scan, progression was one possible future outcome represented in the model's probability distribution.

The new CT then supplied actual new evidence about what had happened, allowing the trajectory to update.
""",
        "Does the lower forecast mean I will progress within six months?",
        "Why did my outlook change?",
    ),
    "Why doesn't OncoTwin just use the latest scan and ignore the old ones?": _q(
        """
Because the meaning of a new scan depends partly on what came before it. The forecasting system was built around longitudinal updates rather than repeatedly treating every scan as if it belonged to a brand-new patient.

Earlier scans provide temporal context for the newest evidence, while the newest scan updates that history.
""",
        "Why does the model use all my previous scans?",
        "Was the model already expecting this progression?",
    ),
    "Does a bigger change in the forecast mean the scan was more severe?": _q(
        """
Not necessarily. The size of the forecast update reflects how the supported new model input changes the model's estimated trajectory given the preceding history. It should not be interpreted as a direct measure of scan "severity" or as a lesion-level severity score.
""",
        "Can you tell which lesion caused the forecast to drop?",
        "What part of the scan mattered most to the model?",
    ),
    "Can you tell which lesion caused the forecast to drop?": _q(
        """
No. OncoTwin should not make lesion-level causal claims from this forecast. We know the model was updated after structured evidence from the new scan was incorporated, and we know what the radiology report described. We cannot validly assign the quantitative change to one particular lesion.
""",
        "What part of the scan mattered most to the model?",
        "Does a bigger change in the forecast mean the scan was more severe?",
    ),
    "What would make the model change again?": _q(
        """
The trajectory can update when new supported model inputs become available, especially subsequent scan evidence that changes the longitudinal picture. A future scan showing continued progression, stability, or another pattern could therefore produce another update.
""",
        "Why does the model use all my previous scans?",
        "What information would be useful next?",
    ),
}


ESR1: dict[str, DemoAnswer] = {
    "What does this new blood test mean?": _q(
        """
The new ctDNA blood test adds molecular information to your cancer history. It detected an ESR1 D538G alteration in circulating tumor DNA. Because your cancer is estrogen-receptor positive, an ESR1 finding can be clinically relevant when your oncology team interprets the biology of the disease and discusses future options.

The result is important patient-state information even though the current frozen forecasting model does not convert this individual result into a new genomic model input.
""",
        "What does ESR1 D538G mean?",
        "Does the ESR1 result change my OncoTwin forecast?",
        "Why is this result important?",
        source_fact_types=("genomic_alteration", "genomic_assay"),
    ),
    "What does ESR1 D538G mean?": _q(
        """
ESR1 is the gene that encodes the estrogen receptor. Your ctDNA result detected an ESR1 D538G alteration in circulating tumor DNA. In ER-positive metastatic breast cancer, ESR1 alterations can provide clinically relevant information about the biology of the cancer and may matter when your oncology team considers future treatment options.
""",
        "Why is this result important?",
        "Could this result affect what treatments my doctor discusses with me?",
        source_fact_types=("genomic_alteration", "genomic_assay", "er_status"),
    ),
    "Why is this result important?": _q(
        """
This result adds new molecular information that was not available from the earlier records. Because your cancer is estrogen-receptor positive, an ESR1 finding can be relevant to discussions about endocrine therapy and future treatment strategy.

OncoTwin records and explains the result, but it does not decide which therapy you should receive.
""",
        "Does the ESR1 result change my OncoTwin forecast?",
        "What should I ask my oncologist about ESR1?",
        source_fact_types=("genomic_alteration", "er_status"),
    ),
    "Did I have this mutation from the beginning?": _q(
        """
The records currently available to OncoTwin do not establish that. The alteration was detected in the later ctDNA test. A later detection does not by itself tell us exactly when the alteration first arose or whether it was present earlier below the level of detection.
""",
        "Why is this result important?",
        source_fact_types=("genomic_alteration", "genomic_assay"),
    ),
    "Does the ESR1 result change my OncoTwin forecast?": _q(
        """
Not in the current deployed forecasting model. The ESR1 result is stored as important new information in your cancer history, but OncoTwin cannot faithfully reconstruct the genomic representation used by the frozen forecasting model from this individual clinical result.

Rather than inventing a genomic input, the forecasting adapter treats genomics as unavailable for this purpose.
""",
        "So why show the ESR1 result if it doesn't change the forecast?",
        "Why can AI explain this result but the prediction model can't use it?",
        source_fact_types=("genomic_alteration", "genomic_assay"),
    ),
    "So why show the ESR1 result if it doesn't change the forecast?": _q(
        """
Because the quantitative forecast is only one part of your cancer state. The ESR1 result can still be clinically meaningful, useful for understanding your disease, relevant to conversations with your oncology team, and relevant to research or trial discussions.

OncoTwin intentionally separates what is clinically important from what the deployed forecasting model can validly consume.
""",
        "Could this result affect what treatments my doctor discusses with me?",
        "Are there clinical trials that might be relevant?",
        source_fact_types=("genomic_alteration",),
    ),
    "Could this result affect what treatments my doctor discusses with me?": _q(
        """
It may be relevant to treatment discussions because ESR1 alterations can matter in ER-positive metastatic breast cancer. Which options are appropriate for you depends on the complete clinical situation.

OncoTwin should help you understand the result and prepare questions rather than tell you which treatment to choose.
""",
        "What should I ask my oncologist about ESR1?",
        "Are there clinical trials that might be relevant?",
        source_fact_types=("genomic_alteration", "er_status"),
    ),
    "What should I ask my oncologist about ESR1?": _q(
        """
Useful questions include:

1. How does the ESR1 D538G result affect how you interpret my cancer now?
2. Does this result change which next-line treatments are most relevant to discuss?
3. Would you want to confirm anything with tissue testing, or is the ctDNA result sufficient for planning?
4. Are there trials specifically relevant to ESR1-mutated ER-positive metastatic breast cancer?

These are questions to bring to your oncology team, not treatment recommendations from OncoTwin.
""",
        "Are there clinical trials that might be relevant?",
        "Could this result affect what treatments my doctor discusses with me?",
        source_fact_types=("genomic_alteration",),
    ),
    "Does this result explain why my cancer progressed?": _q(
        """
The ESR1 finding is biologically relevant, but this individual result does not by itself prove that it caused the progression seen on your scan. OncoTwin should not convert a plausible biological relationship into a patient-specific causal conclusion that the records do not establish.
""",
        "Did I have this mutation from the beginning?",
        "Why can AI explain this result but the prediction model can't use it?",
        source_fact_types=("genomic_alteration", "scan_assessment"),
    ),
    "What changed after adding this blood test?": _q(
        """
Your patient state became richer. The record now contains a verified ESR1 D538G ctDNA finding that can appear in your molecular summary, appointment preparation, research context, and questions for your oncology team.

The current forecasting system does not manufacture a genomic model input from this finding, so adding the result does not by itself create a new genomics-driven quantitative forecast effect.
""",
        "Does the ESR1 result change my OncoTwin forecast?",
        "So why show the ESR1 result if it doesn't change the forecast?",
        source_fact_types=("genomic_alteration", "genomic_assay"),
    ),
    "Why can AI explain this result but the prediction model can't use it?": _q(
        """
They are different systems with different requirements. The language layer can explain a verified clinical fact and connect it to the rest of your record. The forecasting model can only use inputs that match the representation on which it was trained and validated.

OncoTwin deliberately refuses to manufacture an unsupported genomic embedding simply to make the forecasting model consume the new ESR1 result. That keeps the explanation layer from silently changing the scientific model.
""",
        "Does the ESR1 result change my OncoTwin forecast?",
        "What changed in my record versus what changed in the model?",
        source_fact_types=("genomic_alteration", "genomic_assay"),
    ),
}


# Questions that remain meaningful after the ESR1 result, even though they first
# become available after the progression scan.
ESR1_WITH_PROGRESSION = {**PROGRESSION, **ESR1}


def question_bank(stage: str) -> dict[str, DemoAnswer]:
    stage_map = BASELINE if stage == "baseline" else PROGRESSION if stage == "after_progression" else ESR1_WITH_PROGRESSION
    return {**COMMON, **stage_map}


def normalize_question(question: str) -> str:
    value = re.sub(r"\s+", " ", question.strip()).lower()
    value = re.sub(r"[?!.]+$", "", value).strip()
    return value


def lookup_question(stage: str, question: str) -> tuple[str | None, DemoAnswer | None]:
    wanted = normalize_question(question)
    for authored, row in question_bank(stage).items():
        if normalize_question(authored) == wanted:
            return authored, row
    return None, None


def suggested_questions(stage: str) -> list[str]:
    return list(SUGGESTED_QUESTIONS.get(stage, SUGGESTED_QUESTIONS["baseline"]))


def render_answer(text: str, forecast_refs: dict[str, Any]) -> str:
    values = {
        "PRE_PFS_6M": forecast_refs.get("pre_pfs_6m_display", "not available"),
        "POST_PFS_6M": forecast_refs.get("selected_pfs_6m_display", "not available"),
        "DELTA_6M_PP": forecast_refs.get("delta_6m_pp_display", "not available"),
    }
    return text.format(**values)
