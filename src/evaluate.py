"""
Script para avaliar prompts otimizados com as 5 métricas oficiais do desafio.

Métricas avaliadas (conforme enunciado):
1. Helpfulness  - Utilidade e relevância da resposta
2. Correctness  - Correção factual comparada ao ground truth
3. F1-Score     - Balanceamento entre Precision e Recall
4. Clarity      - Clareza e estrutura da resposta
5. Precision    - Ausência de alucinações e foco na pergunta

Critério de aprovação: TODAS as métricas >= 0.9
"""

import os
import sys
import json
import time
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from langsmith import Client
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from utils import check_env_vars, format_score, print_section_header, get_llm as get_configured_llm
from metrics import (
    evaluate_f1_score,
    evaluate_clarity,
    evaluate_precision,
    evaluate_helpfulness,
    evaluate_correctness,
)

load_dotenv()

MINIMUM_SCORE = 0.9


def get_llm():
    return get_configured_llm(temperature=0)


# ── Dataset ───────────────────────────────────────────────────────────────────

def load_dataset_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    examples = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return examples
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {jsonl_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao parsear JSONL: {e}")
        return []


def create_evaluation_dataset(client: Client, dataset_name: str, jsonl_path: str) -> str:
    print(f"\n📦 Preparando dataset: {dataset_name}")
    examples = load_dataset_from_jsonl(jsonl_path)

    if not examples:
        print("❌ Nenhum exemplo carregado.")
        return dataset_name

    print(f"   ✓ {len(examples)} exemplos carregados de {jsonl_path}")

    try:
        existing = [ds for ds in client.list_datasets(dataset_name=dataset_name)
                    if ds.name == dataset_name]
        if existing:
            print(f"   ✓ Dataset '{dataset_name}' já existe, reutilizando.")
        else:
            dataset = client.create_dataset(dataset_name=dataset_name)
            for ex in examples:
                client.create_example(
                    dataset_id=dataset.id,
                    inputs=ex["inputs"],
                    outputs=ex["outputs"],
                )
            print(f"   ✓ Dataset criado com {len(examples)} exemplos.")
    except Exception as e:
        print(f"   ⚠️  Erro ao criar dataset: {e}")

    return dataset_name


# ── Pull do prompt ────────────────────────────────────────────────────────────

def pull_prompt(prompt_name: str) -> ChatPromptTemplate:
    username = os.getenv("USERNAME_LANGSMITH_HUB", "")
    full_name = f"{username}/{prompt_name}" if username and "/" not in prompt_name else prompt_name

    print(f"   🔗 Puxando prompt: {full_name}")
    try:
        prompt = hub.pull(full_name)
        print(f"   ✓ Prompt carregado com sucesso.")
        return prompt
    except Exception as e:
        print(f"\n❌ Não foi possível carregar o prompt '{full_name}'")
        print(f"   Verifique se o push foi feito: python src/push_prompts.py")
        print(f"   Erro: {e}")
        raise


# ── Avaliação por exemplo ─────────────────────────────────────────────────────

def run_prompt_on_example(
    prompt_template: ChatPromptTemplate,
    example: Any,
    llm: Any,
) -> Dict[str, str]:
    try:
        inputs = example.inputs if hasattr(example, "inputs") else {}
        outputs = example.outputs if hasattr(example, "outputs") else {}

        chain = prompt_template | llm
        response = chain.invoke(inputs)
        answer = response.content

        question = inputs.get("bug_report", "") if isinstance(inputs, dict) else ""
        reference = outputs.get("reference", "") if isinstance(outputs, dict) else ""

        return {"question": question, "answer": answer, "reference": reference}

    except Exception as e:
        print(f"      ⚠️  Erro ao executar exemplo: {e}")
        return {"question": "", "answer": "", "reference": ""}


def evaluate_example(result: Dict[str, str]) -> Dict[str, float]:
    """Calcula as 5 métricas oficiais para um único exemplo."""
    q = result["question"]
    a = result["answer"]
    r = result["reference"]

    return {
        "helpfulness": evaluate_helpfulness(q, a, r)["score"],
        "correctness": evaluate_correctness(q, a, r)["score"],
        "f1_score":    evaluate_f1_score(q, a, r)["score"],
        "clarity":     evaluate_clarity(q, a, r)["score"],
        "precision":   evaluate_precision(q, a, r)["score"],
    }


# ── Avaliação do prompt completo ──────────────────────────────────────────────

def evaluate_prompt(
    prompt_name: str,
    dataset_name: str,
    client: Client,
) -> Dict[str, float]:
    print(f"\n🔍 Avaliando prompt: {prompt_name}")

    try:
        prompt_template = pull_prompt(prompt_name)
    except Exception:
        return {k: 0.0 for k in ["helpfulness", "correctness", "f1_score", "clarity", "precision"]}

    examples = list(client.list_examples(dataset_name=dataset_name))
    print(f"   📋 Dataset: {len(examples)} exemplos")

    llm = get_llm()

    all_scores: Dict[str, list] = {
        "helpfulness": [], "correctness": [], "f1_score": [],
        "clarity": [], "precision": [],
    }

    print("   ⏳ Avaliando exemplos...")

    for i, example in enumerate(examples, 1):
        result = run_prompt_on_example(prompt_template, example, llm)

        if not result["answer"]:
            print(f"      [{i}/{len(examples)}] ⚠️  Resposta vazia, pulando.")
            continue

        scores = evaluate_example(result)

        for key in all_scores:
            all_scores[key].append(scores[key])

        print(
            f"      [{i}/{len(examples)}] "
            f"Help:{scores['helpfulness']:.2f} "
            f"Corr:{scores['correctness']:.2f} "
            f"F1:{scores['f1_score']:.2f} "
            f"Clar:{scores['clarity']:.2f} "
            f"Prec:{scores['precision']:.2f}"
        )

        if i < len(examples):
            time.sleep(1)

    return {
        key: round(sum(vals) / len(vals), 4) if vals else 0.0
        for key, vals in all_scores.items()
    }


# ── Exibição de resultados ────────────────────────────────────────────────────

def display_results(prompt_name: str, scores: Dict[str, float]) -> bool:
    print("\n" + "=" * 50)
    print(f"Prompt: {prompt_name}")
    print("=" * 50)

    metric_labels = {
        "helpfulness": "Helpfulness",
        "correctness": "Correctness",
        "f1_score":    "F1-Score",
        "clarity":     "Clarity",
        "precision":   "Precision",
    }

    all_passed = True
    for key, label in metric_labels.items():
        score = scores.get(key, 0.0)
        if score < MINIMUM_SCORE:
            all_passed = False
        print(f"  - {label}: {format_score(score, threshold=MINIMUM_SCORE)}")

    average = sum(scores.values()) / len(scores) if scores else 0.0
    print("\n" + "-" * 50)
    print(f"📊 MÉDIA GERAL: {average:.4f}")
    print("-" * 50)

    if all_passed:
        print(f"\n✅ STATUS: APROVADO — todas as métricas >= {MINIMUM_SCORE}")
    else:
        failed = [metric_labels[k] for k, v in scores.items() if v < MINIMUM_SCORE]
        print(f"\n❌ STATUS: REPROVADO")
        print(f"   Métricas abaixo de {MINIMUM_SCORE}: {', '.join(failed)}")

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print_section_header("AVALIAÇÃO DE PROMPTS — DESAFIO MBA IA")

    provider   = os.getenv("LLM_PROVIDER", "openai")
    llm_model  = os.getenv("LLM_MODEL", "gpt-4o-mini")
    eval_model = os.getenv("EVAL_MODEL", "gpt-4o")
    username   = os.getenv("USERNAME_LANGSMITH_HUB", "")

    print(f"Provider:           {provider}")
    print(f"Modelo principal:   {llm_model}")
    print(f"Modelo avaliação:   {eval_model}")
    print(f"LangSmith username: {username or '⚠️  não configurado'}")

    required_vars = ["LANGSMITH_API_KEY", "USERNAME_LANGSMITH_HUB", "LLM_PROVIDER"]
    if provider == "openai":
        required_vars.append("OPENAI_API_KEY")
    elif provider in ["google", "gemini"]:
        required_vars.append("GOOGLE_API_KEY")

    if not check_env_vars(required_vars):
        return 1

    jsonl_path = "datasets/bug_to_user_story.jsonl"
    if not Path(jsonl_path).exists():
        print(f"\n❌ Dataset não encontrado: {jsonl_path}")
        return 1

    client = Client()
    project_name = os.getenv("LANGSMITH_PROJECT", "evaluation-prompt-project")
    dataset_name = f"{project_name}-eval"
    create_evaluation_dataset(client, dataset_name, jsonl_path)

    prompts_to_evaluate = ["bug_to_user_story_v2"]
    results_summary = []
    all_passed = True

    for prompt_name in prompts_to_evaluate:
        try:
            scores = evaluate_prompt(prompt_name, dataset_name, client)
            passed = display_results(prompt_name, scores)
            all_passed = all_passed and passed
            results_summary.append({"prompt": prompt_name, "scores": scores, "passed": passed})
        except Exception as e:
            print(f"\n❌ Falha ao avaliar '{prompt_name}': {e}")
            all_passed = False
            results_summary.append({
                "prompt": prompt_name,
                "scores": {k: 0.0 for k in ["helpfulness", "correctness", "f1_score", "clarity", "precision"]},
                "passed": False,
            })

    print("\n" + "=" * 50)
    print("RESUMO FINAL")
    print("=" * 50)
    print(f"Prompts avaliados: {len(results_summary)}")
    print(f"Aprovados:         {sum(1 for r in results_summary if r['passed'])}")
    print(f"Reprovados:        {sum(1 for r in results_summary if not r['passed'])}")

    if all_passed:
        print(f"\n✅ Todos os prompts aprovados com todas as métricas >= {MINIMUM_SCORE}!")
        print(f"\n🔗 Dashboard: https://smith.langchain.com/projects/{project_name}")
        return 0
    else:
        print(f"\n⚠️  Nem todos os prompts atingiram {MINIMUM_SCORE} em todas as métricas.")
        return 1


if __name__ == "__main__":
    sys.exit(main())