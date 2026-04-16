"""
Script para fazer pull de prompts do LangSmith Prompt Hub.

Este script:
1. Conecta ao LangSmith usando credenciais do .env
2. Faz pull dos prompts do Hub (leonanluppi/bug_to_user_story_v1)
3. Salva localmente em prompts/bug_to_user_story_v1.yml
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from utils import save_yaml, check_env_vars, print_section_header

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

PROMPTS_TO_PULL = [
    {
        "hub_name": "leonanluppi/bug_to_user_story_v1",
        "local_filename": "bug_to_user_story_v1.yml",
    }
]

REQUIRED_ENV_VARS = ["LANGCHAIN_API_KEY", "LANGCHAIN_ENDPOINT"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_messages_from_prompt(prompt) -> list[dict]:
    """
    Extrai as mensagens de um ChatPromptTemplate do LangChain
    e devolve uma lista de dicts { role, content }.
    Funciona com HumanMessagePromptTemplate, SystemMessagePromptTemplate
    e qualquer outro tipo que exponha .prompt.template.
    """
    messages = []

    for msg in prompt.messages:
        # Determina o role a partir do tipo da mensagem
        class_name = type(msg).__name__.lower()
        if "system" in class_name:
            role = "system"
        elif "human" in class_name or "user" in class_name:
            role = "human"
        elif "ai" in class_name or "assistant" in class_name:
            role = "ai"
        else:
            role = "generic"

        # Extrai o template de texto
        if hasattr(msg, "prompt") and hasattr(msg.prompt, "template"):
            content = msg.prompt.template
        elif hasattr(msg, "content"):
            content = msg.content
        else:
            content = str(msg)

        messages.append({"role": role, "content": content})

    return messages


def build_yaml_payload(prompt, hub_name: str) -> dict:
    """
    Monta o dicionário que será salvo como YAML.
    Inclui metadados e a lista de mensagens do prompt.
    """
    messages = extract_messages_from_prompt(prompt)

    # Coleta input_variables (variáveis entre {chaves} no template)
    input_variables = list(prompt.input_variables) if hasattr(prompt, "input_variables") else []

    return {
        "name": hub_name,
        "version": "v1",
        "description": "Prompt original (baixa qualidade) extraído do LangSmith Prompt Hub.",
        "input_variables": input_variables,
        "messages": messages,
        "metadata": {
            "source": f"https://smith.langchain.com/hub/{hub_name}",
            "quality": "low",
            "techniques": [],
        },
    }


# ── Core ─────────────────────────────────────────────────────────────────────

def pull_prompts_from_langsmith() -> bool:
    """
    Faz pull de cada prompt listado em PROMPTS_TO_PULL,
    converte para YAML e salva em PROMPTS_DIR.

    Retorna True se todos os pulls tiverem sucesso, False caso contrário.
    """
    print_section_header("Pull de Prompts do LangSmith")

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for item in PROMPTS_TO_PULL:
        hub_name = item["hub_name"]
        local_filename = item["local_filename"]
        dest_path = PROMPTS_DIR / local_filename

        print(f"\n📥  Fazendo pull: {hub_name}")

        try:
            prompt = hub.pull(hub_name)
            print(f"    ✅  Pull concluído — tipo: {type(prompt).__name__}")
        except Exception as exc:
            print(f"    ❌  Falha ao fazer pull de '{hub_name}': {exc}")
            all_ok = False
            continue

        try:
            payload = build_yaml_payload(prompt, hub_name)
            save_yaml(payload, dest_path)
            print(f"    💾  Salvo em: {dest_path}")
        except Exception as exc:
            print(f"    ❌  Falha ao salvar '{local_filename}': {exc}")
            all_ok = False

    return all_ok


# ── Entry-point ───────────────────────────────────────────────────────────────

def main() -> int:
    """Ponto de entrada do script. Retorna 0 em sucesso, 1 em falha."""

    # 1. Verifica variáveis de ambiente obrigatórias
    missing = check_env_vars(REQUIRED_ENV_VARS)
    if missing:
        print(f"❌  Variáveis de ambiente ausentes: {', '.join(missing)}")
        print("    Configure o arquivo .env com base no .env.example e tente novamente.")
        return 1

    # 2. Executa o pull
    success = pull_prompts_from_langsmith()

    if success:
        print("\n✅  Todos os prompts foram baixados com sucesso!")
        print(f"    Verifique a pasta: {PROMPTS_DIR.resolve()}")
        return 0
    else:
        print("\n⚠️   Um ou mais prompts falharam. Verifique os erros acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())