import gradio as gr
import numpy as np
from transformers import pipeline
from transformers.trainer_utils import get_last_checkpoint
import json

print(
    json.dumps(
        """The CEDMO hub’s fact-checking activities are based on an experienced and extensive ecosystem of fact-checkers, disinformation analysts, media literacy organisations and academics who detect, analyse, and expose emerging harmful information disorders. Therefore, the project pays special attention to disinformation targeting Central European and EU issues and policies. Through a rapid-alerts network, fact-checking and investigation, reports are sent to the relevant target group (media, public institutions, civil society and government) to minimise the impact of disinformation campaigns. Immediate disinformation responses and daily fact checking is delivered by seasoned professionals of the international news agency AFP , Demagog.cz, Demagog.sk, Konkret24 and Infosecurity.sk .""",
        ensure_ascii=False,
    )
)

MODELS = {
    "🇨🇿 MBart (SumeCzech)": "ctu-aic/mbart-sumeczech-claim-extraction",
    "🇨🇿 MBart": "ctu-aic/mbart25-large-eos",
    "🇬🇧 T5-small (BBC)": "ctu-aic/t5-small-feversum",
    "🇬🇧 T5-large (CNN)": "ctu-aic/t5-large-feversum",
    # "🇬🇧 Pegasus (BBC)": "/home/ullriher/ullriher/models/promising/t5-large-finetuned-xsum-cnn_feversum3_text2claim_bs2_ep30",
    "🇸🇰 MBart (SlovakSum)": "ctu-aic/mbart-slovaksum",
}


def get_pipeline(model_name_or_path):
    try:
        return pipeline("summarization", model=model_name_or_path, device="cuda:0")
    except:
        # return pipeline("summarization", model=get_last_checkpoint(model_name_or_path), device="cuda:0")
        return None


summarizers = {}
for name, model in MODELS.items():
    summarizer = get_pipeline(model)
    if summarizer is not None:
        summarizers[name] = summarizer


def output_to_text(output):
    return output[0]["summary_text"].replace("cs ", "").replace("<pad>", "").replace("</s>", "")


def process(input, summarizer, claims=1, k=1, min_length=10, max_length=40):
    output = [
        output_to_text(
            summarizers[summarizer](
                input,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
        )
    ]

    for _ in range(claims - 1):
        output.append(
            output_to_text(
                summarizers[summarizer](
                    input, max_length=max_length, min_length=min_length, do_sample=True, top_k=k
                )
            )
        )

    return output


LONG_TEXT = "Prodej živých delfínů se řídí dohodou o mezinárodním obchodu s ohroženými druhy, která zakazuje podobné transakce, pokud by mohly zvířatům uškodit. Šalamounovy ostrovy, ležící asi 1800 kilometrů severovýchodně od Austrálie, nicméně dohodu nepodepsaly. Území je v současné době zmítáno politickou krizí a etnickými násilnostmi, kvůli nimž sem byli tento týden vysláni australští vojáci. Ekologové viní mexické podnikatele, že krize na Šalamounových ostrovech zneužili."

class MyInterface(gr.Interface):
    def __init__(self):
        gr.Interface.__init__(
            self,
            process,
            title="Factual Claim Extraction",
            description="This is a prototype CEDMO application to extract factual claims from an arbitrary text.",
            inputs=[
                gr.inputs.Textbox(lines=5, label="Text to extract"),
                gr.inputs.Radio(list(summarizers.keys()), label="Model", default="🇨🇿 MBart"),
                gr.Slider(1, 10, step=1, label="Number of claims"),
                gr.Slider(1, 100, 10, step=1, label="Amount of randomness"),
                gr.Slider(1, 100, 10, step=1, label="Min length (# tokens)"),
                gr.Slider(1, 100, 40, step=1, label="Max length (# tokens)"),
            ],
            outputs=[gr.components.JSON(label="Claims")],
            theme=gr.themes.Soft(primary_hue="yellow"),
        )

    def render_title_description(self) -> None:
        if self.title:
            gr.Markdown(
                "<h1 style='text-align: left; margin-bottom: .5rem;color:#3c3950'>"
                + '<img src="https://cedmohub.eu/wp-content/uploads/thegem-logos/logo_97ce70140f90745805929b382597e9b5_2x.png" style="height: 2.5rem; margin-right: 1.5rem; vertical-align: middle; float:left;"/>'
                + self.title
                + "</h1>"
            )
        if self.description:
            gr.Markdown(self.description)


demo = MyInterface()

if __name__ == "__main__":
    demo.launch(share=True, inbrowser=True)
