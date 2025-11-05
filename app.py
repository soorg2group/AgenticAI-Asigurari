import os
import gradio as gr
from openai import OpenAI

# Supabase (opțional, pentru context local)
try:
    from supabase import create_client
except Exception:
    create_client = None

# Secrete din Hugging Face Spaces -> Settings -> Variables/Secrets
PPLX_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY", "")

# Client OpenAI-compat către Perplexity (Chat Completions)
client = OpenAI(api_key=PPLX_API_KEY, base_url="https://api.perplexity.ai")

supabase = None
if SUPABASE_URL and SUPABASE_KEY and create_client is not None:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = (
    "Ești un agent AI pentru un broker de asigurări din România.\n"
    "- Prioritizează căutarea în baza de cunoștințe (Context local) și folosește informațiile găsite.\n"
    "- Răspunde concis în română (120–180 cuvinte) și explică pe scurt: acoperiri, excluderi, fransize, limite, pașii următori.\n"
    "- Nu oferi consultanță juridică sau promisiuni; adaugă un scurt disclaimer și propune handoff la consultant uman când e necesar.\n"
    "- Dacă lipsește contextul local sau informația e incertă, comunică explicit și cere clarificări.\n"
    "- Respectă confidențialitatea: nu solicita CNP/IBAN/date card.\n"
    "- Dacă întrebarea depășește domeniul asigurărilor, spune politicos și redirecționează.\n"
    "- Ton profesionist, empatic, fără jargon inutil; folosește liste scurte când ajută.\n"
    "- Pentru tabele sau date multiple, structurează clar; pentru date/termen, fii precis.\n"
    "- Menține contextul conversației și evită repetarea inutilă."
)

def fetch_context_from_supabase(query: str, limit: int = 3) -> str:
    """
    Citește câteva fragmente din tabela 'kb_chunks' (coloane: id, text, tags).
    Funcționează fără vector search; e suficient pentru demo inițial low-cost.
    """
    if supabase is None:
        return ""
    try:
        # Select simplu, limitat (poți înlocui ulterior cu un filtru după tags)
        resp = supabase.table("kb_chunks").select("text").limit(limit).execute()
        data = getattr(resp, "data", None)
        if data:
            chunks = [row.get("text", "") for row in data if row.get("text")]
            return "\n\n".join(chunks)[:1500]
    except Exception:
        return ""
    return ""

def answer_fn(message, history, rasp_len):
    # Controlează costul prin max_tokens în funcție de selectarea utilizatorului
    max_tokens = 180 if rasp_len == "scurt" else 280

    kb_context = fetch_context_from_supabase(message, limit=3)
    user_prompt = ""
    if kb_context:
        user_prompt += f"Context local (FAQ/condiții):\n{kb_context}\n\n"
    user_prompt += f"Întrebare utilizator: {message}\n"

    try:
        completion = client.chat.completions.create(
            model="sonar",  # model Perplexity ieftin (web-grounded)
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        answer = f"Eroare la interogarea modelului: {e}\nVerifică PERPLEXITY_API_KEY și modelul."
    return answer

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Agent BrokerAI Asigurări (proto)")
    gr.Markdown("Chat demo BrokerAI.sorinolariu.org + context optional Supabase. Momentan nu ofera consultanta juridica.")
    len_radio = gr.Radio(choices=["scurt", "mediu"], value="scurt", label="Răspuns")
    chat = gr.ChatInterface(
        fn=lambda message, history: answer_fn(message, history, len_radio.value),
        textbox=gr.Textbox(placeholder="Scrie întrebarea ta despre RCA/CASCO/Locuință/Viață..."),
        title=None,
        description=None,
        submit_btn="Trimite",
        retry_btn="Regenerează",
        undo_btn="Înapoi",
        clear_btn="Curăță"
    )

if __name__ == "__main__":
    demo.launch()
