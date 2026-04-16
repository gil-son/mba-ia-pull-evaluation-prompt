"""
Script para fazer push de prompts otimizados ao LangSmith Prompt Hub.

Este script:
1. Lê os prompts otimizados de prompts/bug_to_user_story_v2.yml
2. Valida os prompts
3. Faz push PÚBLICO para o LangSmith Hub
4. Adiciona metadados (tags, descrição, técnicas utilizadas)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import Client
from utils import load_yaml, check_env_vars, print_section_header

load_dotenv()

# ── Configurações ────────────────────────────────────────────────────────────
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

PROMPTS_TO_PUSH = [
    {
        "local_filename": "bug_to_user_story_v2.yml",
        "hub_name": "bug_to_user_story_v2",          # {username}/ é prefixado automaticamente
    }
]

REQUIRED_ENV_VARS = ["LANGSMITH_API_KEY", "LANGSMITH_ENDPOINT", "USERNAME_LANGSMITH_HUB"]


# ── Validação ─────────────────────────────────────────────────────────────────

def validate_prompt(prompt_data: dict) -> tuple[bool, list]:
    """
    Valida estrutura básica de um prompt.

    Args:
        prompt_data: Dicionário carregado do YAML.

    Returns:
        (is_valid, errors) — True + lista vazia se válido, False + erros se inválido.
    """
    errors = []

    # Campos obrigatórios de topo
    for field in ("name", "messages", "metadata"):
        if not prompt_data.get(field):
            errors.append(f"Campo obrigatório ausente ou vazio: '{field}'")

    messages = prompt_data.get("messages", [])

    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("'messages' deve ser uma lista não vazia.")
    else:
        # Verifica se há pelo menos um system e um human
        roles = [m.get("role", "") for m in messages]
        if "system" not in roles:
            errors.append("Nenhuma mensagem com role 'system' encontrada.")
        if "human" not in roles:
            errors.append("Nenhuma mensagem com role 'human' encontrada.")

        # Verifica se nenhuma mensagem tem conteúdo vazio
        for i, msg in enumerate(messages):
            if not msg.get("content", "").strip():
                errors.append(f"Mensagem {i} (role={msg.get('role')}) tem conteúdo vazio.")

    # Verifica técnicas nos metadados
    techniques = prompt_data.get("metadata", {}).get("techniques", [])
    if not isinstance(techniques, list) or len(techniques) < 2:
        errors.append("metadata.techniques deve conter pelo menos 2 técnicas.")

    is_valid = len(errors) == 0
    return is_valid, errors


# ── Conversão YAML → ChatPromptTemplate ──────────────────────────────────────

def build_chat_prompt_template(prompt_data: dict) -> ChatPromptTemplate:
    """
    Converte o dicionário do YAML em um ChatPromptTemplate do LangChain.

    Suporta roles: system, human, ai.
    """
    role_map = {
        "system": SystemMessagePromptTemplate,
        "human": HumanMessagePromptTemplate,
    }

    lc_messages = []
    for msg in prompt_data["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role in role_map:
            lc_messages.append(role_map[role].from_template(content))
        else:
            # Fallback: mensagem genérica
            lc_messages.append(HumanMessagePromptTemplate.from_template(content))

    return ChatPromptTemplate.from_messages(lc_messages)


# ── Push ──────────────────────────────────────────────────────────────────────

def push_prompt_to_langsmith(prompt_name: str, prompt_data: dict) -> bool:
    """
    Faz push do prompt otimizado para o LangSmith Hub (PÚBLICO).

    Args:
        prompt_name: Nome do prompt no Hub (sem o prefixo de username).
        prompt_data: Dicionário com os dados do prompt (carregado do YAML).

    Returns:
        True se o push foi bem-sucedido, False caso contrário.
    """
    username = os.getenv("USERNAME_LANGSMITH_HUB")
    full_hub_name = f"{username}/{prompt_name}"

    print(f"\n📤  Fazendo push: {full_hub_name}")

    try:
        # 1. Monta o ChatPromptTemplate
        chat_prompt = build_chat_prompt_template(prompt_data)

        # 2. Extrai metadados para incluir na descrição e tags
        metadata = prompt_data.get("metadata", {})
        techniques = metadata.get("techniques", [])
        description = prompt_data.get("description", "Prompt otimizado.")

        tags = ["optimized", "bug-to-user-story"] + [
            t.lower().replace(" ", "-") for t in techniques
        ]

        # 3. Push para o Hub — new_repo_is_public=True garante visibilidade pública
        hub.push(
            full_hub_name,
            chat_prompt,
            new_repo_description=description,
            new_repo_is_public=True,
            tags=tags,
        )

        print(f"    ✅  Push concluído!")
        print(f"    🔗  URL: https://smith.langchain.com/hub/{full_hub_name}")
        print(f"    🏷️   Tags: {', '.join(tags)}")
        return True

    except Exception as exc:
        print(f"    ❌  Falha no push de '{full_hub_name}': {exc}")
        return False


# ── Entry-point ───────────────────────────────────────────────────────────────

def check_missing_vars(required: list) -> list:
    """Verifica variáveis ausentes diretamente, sem depender do retorno de check_env_vars."""
    return [var for var in required if not os.getenv(var)]


def main() -> int:
    """Ponto de entrada. Retorna 0 em sucesso, 1 em falha."""

    # 1. Verifica variáveis de ambiente diretamente
    missing = check_missing_vars(REQUIRED_ENV_VARS)
    if missing:
        print(f"❌  Variáveis de ambiente ausentes: {', '.join(missing)}")
        print("    Adicione-as ao .env e tente novamente.")
        print("    Necessário: LANGSMITH_API_KEY, LANGSMITH_ENDPOINT, USERNAME_LANGSMITH_HUB")
        return 1

    print_section_header("Push de Prompts Otimizados ao LangSmith")

    all_ok = True

    for item in PROMPTS_TO_PUSH:
        local_filename = item["local_filename"]
        hub_name = item["hub_name"]
        prompt_path = PROMPTS_DIR / local_filename

        print(f"\n📂  Carregando: {prompt_path}")

        # 2. Carrega o YAML
        if not prompt_path.exists():
            print(f"    ❌  Arquivo não encontrado: {prompt_path}")
            print(f"    ⚠️   Execute primeiro: python src/pull_prompts.py")
            all_ok = False
            continue

        try:
            prompt_data = load_yaml(prompt_path)
        except Exception as exc:
            print(f"    ❌  Erro ao ler YAML '{local_filename}': {exc}")
            all_ok = False
            continue

        # 3. Valida o prompt
        is_valid, errors = validate_prompt(prompt_data)
        if not is_valid:
            print(f"    ❌  Prompt inválido — erros encontrados:")
            for err in errors:
                print(f"        • {err}")
            all_ok = False
            continue

        print(f"    ✅  Validação OK")

        # 4. Faz o push
        success = push_prompt_to_langsmith(hub_name, prompt_data)
        if not success:
            all_ok = False

    # 5. Resultado final
    if all_ok:
        print("\n✅  Todos os prompts foram publicados com sucesso no LangSmith!")
        username = os.getenv("USERNAME_LANGSMITH_HUB")
        print(f"    Dashboard: https://smith.langchain.com/hub/{username}")
        return 0
    else:
        print("\n⚠️   Um ou mais prompts falharam. Verifique os erros acima.")
        return 1


if __name__ == "__main__":
    sys.exit(main())